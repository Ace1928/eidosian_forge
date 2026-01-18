import unittest
from Cython.Compiler import PyrexTypes as pt
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.PyrexTypes import CFuncTypeArg
def cppclasstype(name, base_classes):
    return pt.CppClassType(name, None, 'CPP_' + name, base_classes)