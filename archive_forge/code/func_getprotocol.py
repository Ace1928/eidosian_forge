import weakref
import importlib_metadata
from wsme.exc import ClientSideError
def getprotocol(name, **options):
    protocol_class = registered_protocols.get(name)
    if protocol_class is None:
        for entry_point in importlib_metadata.entry_points().select(group='wsme.protocols'):
            if entry_point.name == name:
                protocol_class = entry_point.load()
        if protocol_class is None:
            raise ValueError("Cannot find protocol '%s'" % name)
        registered_protocols[name] = protocol_class
    return protocol_class(**options)