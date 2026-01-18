import sys
from .version import __build__, __version__
def objid(obj):
    return obj.__class__.__name__ + ':' + hex(id(obj))