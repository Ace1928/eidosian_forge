import sys, os
from distutils.core import Command
from distutils.errors import DistutilsOptionError
from distutils.util import get_platform
def show_compilers():
    from distutils.ccompiler import show_compilers
    show_compilers()