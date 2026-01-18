import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def nestedcode(func, recurse=True):
    """get the code objects for any nested functions (e.g. in a closure)"""
    func = code(func)
    if not iscode(func):
        return []
    nested = set()
    for co in func.co_consts:
        if co is None:
            continue
        co = code(co)
        if co:
            nested.add(co)
            if recurse:
                nested |= set(nestedcode(co, recurse=True))
    return list(nested)