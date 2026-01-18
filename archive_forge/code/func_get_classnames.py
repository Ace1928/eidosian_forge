import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
def get_classnames(packname):
    res = methods_env['getClasses'](where=StrSexpVector(('package:%s' % packname,)))
    return tuple(res)