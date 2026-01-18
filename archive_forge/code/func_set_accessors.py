import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
def set_accessors(cls, cls_name, where, acs):
    if where is None:
        where = rinterface.globalenv
    else:
        where = 'package:' + str(where)
        where = StrSexpVector((where,))
    for r_name, python_name, as_property, docstring in acs:
        if python_name is None:
            python_name = r_name
        r_meth = getmethod(StrSexpVector((r_name,)), signature=StrSexpVector((cls_name,)), where=where)
        r_meth = conversion.get_conversion().rpy2py(r_meth)
        if as_property:
            setattr(cls, python_name, property(r_meth, None, None))
        else:
            setattr(cls, python_name, lambda self: r_meth(self))