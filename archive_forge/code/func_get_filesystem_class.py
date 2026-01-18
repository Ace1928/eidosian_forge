from __future__ import annotations
import importlib
import types
import warnings
def get_filesystem_class(protocol):
    """Fetch named protocol implementation from the registry

    The dict ``known_implementations`` maps protocol names to the locations
    of classes implementing the corresponding file-system. When used for the
    first time, appropriate imports will happen and the class will be placed in
    the registry. All subsequent calls will fetch directly from the registry.

    Some protocol implementations require additional dependencies, and so the
    import may fail. In this case, the string in the "err" field of the
    ``known_implementations`` will be given as the error message.
    """
    if not protocol:
        protocol = default
    if protocol not in registry:
        if protocol not in known_implementations:
            raise ValueError(f'Protocol not known: {protocol}')
        bit = known_implementations[protocol]
        try:
            register_implementation(protocol, _import_class(bit['class']))
        except ImportError as e:
            raise ImportError(bit['err']) from e
    cls = registry[protocol]
    if getattr(cls, 'protocol', None) in ('abstract', None):
        cls.protocol = protocol
    return cls