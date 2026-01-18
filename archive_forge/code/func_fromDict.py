from .python3_compat import iterkeys, iteritems, Mapping  #, u
@classmethod
def fromDict(cls, d, default_factory):
    return munchify(d, factory=lambda d_: cls(default_factory, d_))