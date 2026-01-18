import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
@property
def is_opt_max(self):
    """Returns True if the the optimisation level is "max" False
        otherwise."""
    return self._raw_value == 'max'