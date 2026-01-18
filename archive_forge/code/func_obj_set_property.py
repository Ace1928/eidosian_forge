from . import _gi
from ._constants import \
def obj_set_property(self, pspec, value):
    name = pspec.name.replace('-', '_')
    prop = getattr(cls, name, None)
    if prop:
        prop.fset(self, value)