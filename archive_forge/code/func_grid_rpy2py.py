import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
def grid_rpy2py(robj):
    warnings.warn('This function is deprecated.', category=DeprecationWarning)
    pyobj = original_rpy2py(robj)
    if not isinstance(pyobj, robjects.RS4):
        rcls = pyobj.rclass
        if rcls is NULL:
            rcls = (None,)
        try:
            cls = _grid_dict[rcls[0]]
            pyobj = cls(pyobj)
        except KeyError:
            pass
    return pyobj