import abc
from os_ken.lib import stringify
@classmethod
def register_packet_type(cls, cls_, type_):
    """Per-protocol dict-like set method.

        Provided for convenience of protocol implementers.
        Internal use only."""
    cls._TYPES[type_] = cls_