import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def bind_callables(self, pdict):
    for p in self.lr_productions:
        p.bind(pdict)