import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_implies_c(self, names):
    """same as feature_implies() but combining 'names'"""
    if isinstance(names, str):
        names = set((names,))
    else:
        names = set(names)
    return names.union(self.feature_implies(names))