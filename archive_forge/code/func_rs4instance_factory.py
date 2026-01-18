import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
def rs4instance_factory(robj):
    """
    Return an RS4 objects (R objects in the 'S4' class system)
    as a Python object of type inheriting from `robjects.methods.RS4`.

    The types are located in the namespace `robjects.methods.rs4classes`,
    and a dummy type is dynamically created whenever necessary.
    """
    clslist = None
    if len(robj.rclass) > 1:
        raise ValueError('Currently unable to handle more than one class per object')
    for rclsname in robj.rclass:
        rcls = _getclass(rclsname)
        return rcls(robj)
    if clslist is None:
        return robj