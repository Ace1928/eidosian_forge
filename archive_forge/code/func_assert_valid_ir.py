import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def assert_valid_ir(self, mod):
    llvm.parse_assembly(str(mod))