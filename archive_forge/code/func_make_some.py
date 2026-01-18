from __future__ import print_function, absolute_import, division
import unittest
def make_some():
    t = ()
    i = OBJECTS_PER_CONTAINER
    while i:
        t = (Dealloc(i), t)
        i -= 1
    return t