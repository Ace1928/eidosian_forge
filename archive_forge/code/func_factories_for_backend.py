import dns._features
import dns.asyncbackend
def factories_for_backend(backend=None):
    if backend is None:
        backend = dns.asyncbackend.get_default_backend()
    return _async_factories[backend.name()]