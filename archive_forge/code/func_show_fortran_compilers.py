from distutils.core import Command
from numpy.distutils import log
def show_fortran_compilers(_cache=None):
    if _cache:
        return
    elif _cache is None:
        _cache = []
    _cache.append(1)
    from numpy.distutils.fcompiler import show_fcompilers
    import distutils.core
    dist = distutils.core._setup_distribution
    show_fcompilers(dist)