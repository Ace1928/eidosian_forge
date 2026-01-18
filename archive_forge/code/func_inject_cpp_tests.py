import os.path
from os.path import join as pjoin
from pyarrow._pyarrow_cpp_tests import get_cpp_tests
def inject_cpp_tests(ns):
    """
    Inject C++ tests as Python functions into namespace `ns` (a dict).
    """
    for case in get_cpp_tests():

        def wrapper(case=case):
            case()
        wrapper.__name__ = wrapper.__qualname__ = case.name
        wrapper.__module__ = ns['__name__']
        ns[case.name] = wrapper