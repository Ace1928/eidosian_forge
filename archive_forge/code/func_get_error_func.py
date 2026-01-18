import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def get_error_func(self):
    self.error_func = self.pdict.get('p_error')