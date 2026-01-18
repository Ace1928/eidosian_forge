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
def split_where(collection, predicate, to_list):
    """:yaql:splitWhere

    Returns collection divided into list of collections where delimiters are
    values for which predicate returns true. Delimiters are deleted from
    result.

    :signature: collection.splitWhere(predicate)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg predicate: function of one argument to be applied on every
        element. Elements for which predicate returns true are delimiters for
        new list
    :argType predicate: lambda
    :returnType: list

    .. code::

        yaql> [1, 2, 3, 4, 5, 6, 7].splitWhere($ mod 3 = 0)
        [[1, 2], [4, 5], [7]]
    """
    lst = to_list(collection)
    start = 0
    end = 0
    while end < len(lst):
        if predicate(lst[end]):
            yield lst[start:end]
            start = end + 1
        end += 1
    if start != end:
        yield lst[start:end]