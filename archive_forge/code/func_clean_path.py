import os, glob, re, sys
from distutils import sysconfig
def clean_path(path):
    return path if sys.platform != 'win32' else path.replace('\\', '/')