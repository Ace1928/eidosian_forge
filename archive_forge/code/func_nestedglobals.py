import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def nestedglobals(func, recurse=True):
    """get the names of any globals found within func"""
    func = code(func)
    if func is None:
        return list()
    import sys
    from .temp import capture
    CAN_NULL = sys.hexversion >= 51052711
    names = set()
    with capture('stdout') as out:
        dis.dis(func)
    for line in out.getvalue().splitlines():
        if '_GLOBAL' in line:
            name = line.split('(')[-1].split(')')[0]
            if CAN_NULL:
                names.add(name.replace('NULL + ', '').replace(' + NULL', ''))
            else:
                names.add(name)
    for co in getattr(func, 'co_consts', tuple()):
        if co and recurse and iscode(co):
            names.update(nestedglobals(co, recurse=True))
    return list(names)