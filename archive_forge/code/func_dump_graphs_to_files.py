import logging
import os
import tempfile
from enum import Enum
from typing import Callable, cast, Dict, Iterable, List, Set
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def dump_graphs_to_files(graphs: Dict[str, fx.GraphModule], folder: str='') -> str:
    if not folder:
        folder = tempfile.mkdtemp()
    for prefix, gm in graphs.items():
        with open(os.path.join(folder, f'{prefix}.graph'), 'w') as fp:
            fp.write(str(gm))
    logger.warning('Dump graphs to %s', folder)
    return folder