from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def hasCorrectColspan(element):
    return not isLeaf(element) and element.name == 'td' and (element.attributes.get('colspan') == '2')