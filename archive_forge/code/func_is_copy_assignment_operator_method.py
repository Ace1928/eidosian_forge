from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_copy_assignment_operator_method(self):
    """Returnrs True if the cursor refers to a copy-assignment operator.

        A copy-assignment operator `X::operator=` is a non-static,
        non-template member function of _class_ `X` with exactly one
        parameter of type `X`, `X&`, `const X&`, `volatile X&` or `const
        volatile X&`.


        That is, for example, the `operator=` in:

           class Foo {
               bool operator=(const volatile Foo&);
           };

        Is a copy-assignment operator, while the `operator=` in:

           class Bar {
               bool operator=(const int&);
           };

        Is not.
        """
    return conf.lib.clang_CXXMethod_isCopyAssignmentOperator(self)