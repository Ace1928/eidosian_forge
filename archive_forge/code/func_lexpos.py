import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def lexpos(self, n):
    return getattr(self.slice[n], 'lexpos', 0)