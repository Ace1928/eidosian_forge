import warnings
from . import ccompiler
from . import unixccompiler
from .npy_pkg_config import *
def customized_fcompiler(plat=None, compiler=None):
    from numpy.distutils.fcompiler import new_fcompiler
    c = new_fcompiler(plat=plat, compiler=compiler)
    c.customize()
    return c