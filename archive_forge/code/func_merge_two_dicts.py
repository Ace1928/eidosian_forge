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
def merge_two_dicts(sign, power, var1, var2, dict1, dict2):
    sign1, power1 = dict1[var1]
    sign2, power2 = dict2[var2]
    new_sign = sign1 * sign * sign2
    new_power = power1 - power - power2
    for v2, (s2, p2) in dict2.items():
        dict1[v2] = (s2 * new_sign, p2 + new_power)
    return dict1