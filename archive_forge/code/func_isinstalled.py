import os
import typing
import warnings
from types import ModuleType
from warnings import warn
import rpy2.rinterface as rinterface
from . import conversion
from rpy2.robjects.functions import (SignatureTranslatedFunction,
from rpy2.robjects import Environment
from rpy2.robjects.packages_utils import (
import rpy2.robjects.help as rhelp
def isinstalled(self, packagename: str):
    if not isinstance(packagename, rinterface.StrSexpVector):
        rinterface.StrSexpVector((packagename,))
    elif len(packagename) > 1:
        raise ValueError('Only specify one package name at a time.')
    nrows = self.nrows
    lib_results, lib_packname_i = (self.lib_results, self.lib_packname_i)
    for i in range(0 + lib_packname_i * nrows, nrows * (lib_packname_i + 1), 1):
        if lib_results[i] == packagename:
            return True
    return False