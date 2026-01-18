from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def palette_name(name, length):
    """Create a palette name like CubeYF_8"""
    return '{0}_{1}'.format(name, length)