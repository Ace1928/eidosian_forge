import importlib
import logging
import os
import sys
def chop_py_suffix(p):
    for suf in ['.py', '.pyc', '.pyo']:
        if p.endswith(suf):
            return p[:-len(suf)]
    return p