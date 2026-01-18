from __future__ import print_function
import os
import platform
import remove_pyreadline
import setuptools.command.easy_install as easy_install
import setuptools.package_index
import shutil
import sys
def locate_package(name):
    import pkg_resources
    try:
        pkg = setuptools.package_index.get_distribution(name)
    except pkg_resources.DistributionNotFound:
        pkg = None
    return pkg