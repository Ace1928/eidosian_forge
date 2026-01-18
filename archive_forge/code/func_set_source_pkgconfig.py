import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def set_source_pkgconfig(self, module_name, pkgconfig_libs, source, source_extension='.c', **kwds):
    from . import pkgconfig
    if not isinstance(pkgconfig_libs, list):
        raise TypeError('the pkgconfig_libs argument must be a list of package names')
    kwds2 = pkgconfig.flags_from_pkgconfig(pkgconfig_libs)
    pkgconfig.merge_flags(kwds, kwds2)
    self.set_source(module_name, source, source_extension, **kwds)