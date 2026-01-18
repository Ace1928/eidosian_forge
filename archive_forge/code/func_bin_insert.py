import functools
import re
import sys
from Xlib.support import lock
def bin_insert(list, element):
    """bin_insert(list, element)

    Insert ELEMENT into LIST.  LIST must be sorted, and ELEMENT will
    be inserted to that LIST remains sorted.  If LIST already contains
    ELEMENT, it will not be duplicated.

    """
    if not list:
        list.append(element)
        return
    lower = 0
    upper = len(list) - 1
    while lower <= upper:
        center = (lower + upper) // 2
        if element < list[center]:
            upper = center - 1
        elif element > list[center]:
            lower = center + 1
        elif element == list[center]:
            return
    if element < list[upper]:
        list.insert(upper, element)
    elif element > list[upper]:
        list.insert(upper + 1, element)