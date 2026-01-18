from ..utils import DummyObject, requires_backends
def create_optimizer(*args, **kwargs):
    requires_backends(create_optimizer, ['tf'])