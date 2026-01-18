import os, glob, re, sys
from distutils import sysconfig
def get_python_include_path():
    return sysconfig.get_python_inc()