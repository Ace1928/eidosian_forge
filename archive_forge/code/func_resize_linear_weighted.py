from typing import Callable, Optional, TypeVar
from ..config import registry
from ..model import Model
from ..types import Floats2d
def resize_linear_weighted(layer: Model[Floats2d, Floats2d], new_nO, *, fill_defaults=None) -> Model[Floats2d, Floats2d]:
    """Create a resized copy of a layer that has parameters W and b and dimensions nO and nI."""
    assert not layer.layers
    assert not layer.ref_names
    assert not layer.shims
    if layer.has_dim('nO') is None:
        layer.set_dim('nO', new_nO)
        return layer
    elif new_nO == layer.get_dim('nO'):
        return layer
    elif layer.has_dim('nI') is None:
        layer.set_dim('nO', new_nO, force=True)
        return layer
    dims = {name: layer.maybe_get_dim(name) for name in layer.dim_names}
    dims['nO'] = new_nO
    new_layer: Model[Floats2d, Floats2d] = Model(layer.name, layer._func, dims=dims, params={name: None for name in layer.param_names}, init=layer.init, attrs=layer.attrs, refs={}, ops=layer.ops)
    new_layer.initialize()
    for name in layer.param_names:
        if layer.has_param(name):
            filler = 0 if not fill_defaults else fill_defaults.get(name, 0)
            _resize_parameter(name, layer, new_layer, filler=filler)
    return new_layer