from . import _gi
from ._constants import \
def get_pspec_args(self):
    ptype = self.type
    if ptype in (TYPE_INT, TYPE_UINT, TYPE_LONG, TYPE_ULONG, TYPE_INT64, TYPE_UINT64, TYPE_FLOAT, TYPE_DOUBLE):
        args = (self.minimum, self.maximum, self.default)
    elif ptype == TYPE_STRING or ptype == TYPE_BOOLEAN or ptype.is_a(TYPE_ENUM) or ptype.is_a(TYPE_FLAGS) or ptype.is_a(TYPE_VARIANT):
        args = (self.default,)
    elif ptype in (TYPE_PYOBJECT, TYPE_GTYPE):
        args = ()
    elif ptype.is_a(TYPE_OBJECT) or ptype.is_a(TYPE_BOXED):
        args = ()
    else:
        raise NotImplementedError(ptype)
    return (self.type, self.nick, self.blurb) + args + (self.flags,)