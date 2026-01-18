import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.method
@specs.parameter('collection', yaqltypes.Iterable())
@specs.parameter('predicate', yaqltypes.Lambda())
@specs.inject('to_list', yaqltypes.Delegate('to_list', method=True))
def slice_where(collection, predicate, to_list):
    """:yaql:sliceWhere

    Splits collection into lists. Within every list predicate evaluated
    on its items returns the same value while predicate evaluated on the
    items of the adjacent lists returns different values. Returns an iterator
    to lists.

    :signature: collection.sliceWhere(predicate)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg predicate: function of one argument to be applied on every
        element. Elements for which predicate returns true are delimiters for
        new list and are present in new collection as separate collections
    :argType predicate: lambda
    :returnType: iterator

    .. code::

        yaql> [1, 2, 3, 4, 5, 6, 7].sliceWhere($ mod 3 = 0)
        [[1, 2], [3], [4, 5], [6], [7]]
    """
    lst = to_list(collection)
    start = 0
    end = 0
    p1 = utils.NO_VALUE
    while end < len(lst):
        p2 = predicate(lst[end])
        if p2 != p1 and p1 is not utils.NO_VALUE:
            yield lst[start:end]
            start = end
        end += 1
        p1 = p2
    if start != end:
        yield lst[start:end]