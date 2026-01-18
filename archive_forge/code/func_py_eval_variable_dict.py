from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
def py_eval_variable_dict(self):

    def create_dict_entry(var1, val):
        sign, power, var2 = val
        assert sign in [+1, -1]
        p = ''
        if self._N == 2:
            sign *= (-1) ** power
        elif power % self._N:
            p = " * d['u'] ** %d" % (power % self._N)
        if sign == +1:
            return "'%s' : d['%s']%s" % (var1, var2, p)
        else:
            return "'%s' : - d['%s']%s" % (var1, var2, p)
    format_str = '(lambda d: {\n          %s})'
    return format_str % ',\n          '.join([create_dict_entry(key, val) for key, val in list(self.canonical_representative.items()) if not key == 1])