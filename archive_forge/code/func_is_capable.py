import glob
import importlib
import os.path
@classmethod
def is_capable(cls, requested_capability):
    """Returns true if the requested capability is supported by this plugin
        """
    for c in requested_capability:
        if c not in cls.capability:
            return False
    return True