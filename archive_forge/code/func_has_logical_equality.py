import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def has_logical_equality(self, ty):
    while isinstance(ty, ir.PointerType):
        ty = ty.pointee
    return not isinstance(ty, ir.LabelType)