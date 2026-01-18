from ..utils import DummyObject, requires_backends
def shape_list(*args, **kwargs):
    requires_backends(shape_list, ['tf'])