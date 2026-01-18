import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def referredglobals(func, recurse=True, builtin=False):
    """get the names of objects in the global scope referred to by func"""
    return globalvars(func, recurse, builtin).keys()