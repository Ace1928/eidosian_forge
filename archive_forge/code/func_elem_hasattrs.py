import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
def elem_hasattrs(self, elem, attrs):
    return all((attr in elem.attrib for attr in attrs))