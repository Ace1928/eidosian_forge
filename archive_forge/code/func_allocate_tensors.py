import ctypes
import enum
import os
import platform
import sys
import numpy as np
def allocate_tensors(self):
    self._ensure_safe()
    return self._interpreter.AllocateTensors()