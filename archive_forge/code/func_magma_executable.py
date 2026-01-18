from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase
from . import utilities
import snappy
import re
import sys
import tempfile
import subprocess
import shutil
def magma_executable():
    for name in ['magma', 'magma.exe']:
        if shutil.which(name):
            return name
    return None