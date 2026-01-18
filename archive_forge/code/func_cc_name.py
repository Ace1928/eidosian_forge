import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def cc_name(self):
    return self.opt().cc_name