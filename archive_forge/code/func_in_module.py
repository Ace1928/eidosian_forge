from __future__ import annotations
def in_module(a, b):
    """Is a the same module as or a submodule of b?"""
    return a == b or (a != None and b != None and a.startswith(b + '.'))