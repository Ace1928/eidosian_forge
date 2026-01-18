import _collections_abc
import sys as _sys
from itertools import chain as _chain
from itertools import repeat as _repeat
from itertools import starmap as _starmap
from keyword import iskeyword as _iskeyword
from operator import eq as _eq
from operator import itemgetter as _itemgetter
from reprlib import recursive_repr as _recursive_repr
from _weakref import proxy as _proxy
def move_to_end(self, key, last=True):
    """Move an existing element to the end (or beginning if last is false).

        Raise KeyError if the element does not exist.
        """
    link = self.__map[key]
    link_prev = link.prev
    link_next = link.next
    soft_link = link_next.prev
    link_prev.next = link_next
    link_next.prev = link_prev
    root = self.__root
    if last:
        last = root.prev
        link.prev = last
        link.next = root
        root.prev = soft_link
        last.next = link
    else:
        first = root.next
        link.prev = root
        link.next = first
        first.prev = soft_link
        root.next = link