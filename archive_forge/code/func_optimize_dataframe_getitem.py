from __future__ import annotations
import operator
import numpy as np
from dask import config, core
from dask.blockwise import Blockwise, fuse_roots, optimize_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, fuse
from dask.utils import ensure_dict
def optimize_dataframe_getitem(dsk, keys):
    from dask.layers import DataFrameIOLayer
    io_layers = [k for k, v in dsk.layers.items() if isinstance(v, DataFrameIOLayer)]

    def _is_selection(layer):
        if not isinstance(layer, Blockwise):
            return False
        if layer.dsk[layer.output][0] != operator.getitem:
            return False
        return True

    def _kind(layer):
        key, ind = layer.indices[1]
        if ind is None:
            if isinstance(key, (tuple, str, list, np.ndarray)) or np.isscalar(key):
                return 'column-selection'
        return 'row-selection'
    layers = dsk.layers.copy()
    dependencies = dsk.dependencies.copy()
    for io_layer_name in io_layers:
        columns = set()
        if any((layers[io_layer_name].name == x[0] for x in keys if isinstance(x, tuple))):
            continue
        deps = dsk.dependents[io_layer_name]
        if not all((_is_selection(dsk.layers[k]) for k in deps)) or {_kind(dsk.layers[k]) for k in deps} not in ({'column-selection'}, {'column-selection', 'row-selection'}):
            continue
        row_select_layers = {k for k in deps if _kind(dsk.layers[k]) == 'row-selection'}
        col_select_layers = deps - row_select_layers
        if len(row_select_layers) > 1:
            continue

        def _walk_deps(dependents, key, success):
            if key == success:
                return True
            deps = dependents[key]
            if deps:
                return all((_walk_deps(dependents, dep, success) for dep in deps))
            else:
                return False
        if row_select_layers:
            row_select_layer = row_select_layers.pop()
            if len(dsk.dependents[row_select_layer]) != 1:
                continue
            _layer = dsk.layers[list(dsk.dependents[row_select_layer])[0]]
            if _is_selection(_layer) and _kind(_layer) == 'column-selection':
                selection = _layer.indices[1][0]
                columns |= set(selection if isinstance(selection, list) else [selection])
            else:
                continue
            if not all((_walk_deps(dsk.dependents, col_select_layer, col_select_layer) for col_select_layer in col_select_layers)):
                continue
        for col_select_layer in col_select_layers:
            selection = dsk.layers[col_select_layer].indices[1][0]
            columns |= set(selection if isinstance(selection, list) else [selection])
        update_blocks = {dep: dsk.layers[dep] for dep in deps}
        old = layers[io_layer_name]
        new = old.project_columns(columns)
        if new.name != old.name:
            assert len(update_blocks)
            for block_key, block in update_blocks.items():
                new_indices = ((new.name, block.indices[0][1]), block.indices[1])
                numblocks = {new.name: block.numblocks[old.name]}
                new_block = Blockwise(block.output, block.output_indices, block.dsk, new_indices, numblocks, block.concatenate, block.new_axes)
                layers[block_key] = new_block
                dependencies[block_key] = {new.name}
            dependencies[new.name] = dependencies.pop(io_layer_name)
        layers[new.name] = new
        if new.name != old.name:
            del layers[old.name]
    new_hlg = HighLevelGraph(layers, dependencies)
    return new_hlg