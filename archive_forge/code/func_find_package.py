import os, glob, re, sys
from distutils import sysconfig
def find_package(which_package):
    if which_package == Package.shiboken2_module:
        return find_shiboken2_module()
    if which_package == Package.shiboken2_generator:
        return find_shiboken2_generator()
    if which_package == Package.pyside2:
        return find_pyside2()
    return None