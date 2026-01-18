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
def equations_as_dicts(self, with_non_zero=True):
    if with_non_zero:
        equations = self.equations_with_non_zero_condition
        variables = self.variables + ['t']
    else:
        equations = self.equations
        variables = self.variables
    result = []
    for f in equations:
        result.append({tuple((m.degree(v) for v in variables)): m.get_coefficient() for m in f.get_monomials()})
    return result