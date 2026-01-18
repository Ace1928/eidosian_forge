import sys
import os
import re
import warnings
import types
import unicodedata
def attlist(self):
    attlist = list(self.non_default_attributes().items())
    attlist.sort()
    return attlist