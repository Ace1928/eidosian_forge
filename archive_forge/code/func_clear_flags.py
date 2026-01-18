import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def clear_flags(self):
    """Reset all flags to zero"""
    for flag in self.flags:
        self.flags[flag] = 0