from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def get_init_multipliers(self, invec):
    self.ASLib.EXTERNAL_AmplInterface_get_init_multipliers(self._obj, invec, len(invec))