from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def set_gnu_getopt(self, gnu_getopt=True):
    """Sets whether or not to use GNU style scanning.

    GNU style allows mixing of flag and non-flag arguments. See
    http://docs.python.org/library/getopt.html#getopt.gnu_getopt

    Args:
      gnu_getopt: bool, whether or not to use GNU style scanning.
    """
    self.__dict__['__use_gnu_getopt'] = gnu_getopt
    self.__dict__['__use_gnu_getopt_explicitly_set'] = True