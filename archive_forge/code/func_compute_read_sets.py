import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def compute_read_sets(self, C, ntrans, nullable):
    FP = lambda x: self.dr_relation(C, x, nullable)
    R = lambda x: self.reads_relation(C, x, nullable)
    F = digraph(ntrans, R, FP)
    return F