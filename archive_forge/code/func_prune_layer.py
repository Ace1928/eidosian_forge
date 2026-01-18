from ..utils import DummyObject, requires_backends
def prune_layer(*args, **kwargs):
    requires_backends(prune_layer, ['torch'])