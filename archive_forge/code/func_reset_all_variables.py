import ctypes
import enum
import os
import platform
import sys
import numpy as np
def reset_all_variables(self):
    return self._interpreter.ResetVariableTensors()