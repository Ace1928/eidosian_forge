import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def floatEqual(x, y, epsilon=1e-08):
    return abs(x - y) < epsilon