from . import _gi
from ._constants import \
def obj_get_property(self, pspec):
    name = pspec.name.replace('-', '_')
    return getattr(self, name, None)