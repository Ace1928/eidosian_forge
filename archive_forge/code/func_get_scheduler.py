from ..utils import DummyObject, requires_backends
def get_scheduler(*args, **kwargs):
    requires_backends(get_scheduler, ['torch'])