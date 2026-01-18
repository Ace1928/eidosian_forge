from __future__ import absolute_import, print_function, division
import os
import sys
import pytest
from petl.compat import izip_longest
def ieq2(expect, actual, cast=None):
    """Test when iterables values are equals twice looking for side effects"""
    ieq(expect, actual, cast)
    ieq(expect, actual, cast)