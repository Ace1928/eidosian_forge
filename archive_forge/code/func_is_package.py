from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def is_package(self, fullname):
    """
        Return true, if the named module is a package.

        We need this method to get correct spec objects with
        Python 3.4 (see PEP451)
        """
    return hasattr(self.__get_module(fullname), '__path__')