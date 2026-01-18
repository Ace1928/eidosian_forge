import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def compute_follow_sets(self, ntrans, readsets, inclsets):
    FP = lambda x: readsets[x]
    R = lambda x: inclsets.get(x, [])
    F = digraph(ntrans, R, FP)
    return F