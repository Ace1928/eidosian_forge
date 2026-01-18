from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
def onlyExisting(func):
    """Returns a filter func that when called with a list,
    only calls func on the non-NotImplemented items of the list,
    and only so if there's at least one item remaining.
    Otherwise returns NotImplemented."""

    def wrapper(lst):
        items = [item for item in lst if item is not NotImplemented]
        return func(items) if items else NotImplemented
    return wrapper