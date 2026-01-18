import os, glob, re, sys
from distutils import sysconfig
def find_pyside2():
    return find_package_path('PySide2')