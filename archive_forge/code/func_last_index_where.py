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
def last_index_where(collection, predicate):
    """:yaql:lastIndexWhere

    Returns the index in the collection of the last item which value
    satisfies the predicate. -1 is a return value if there is no such item

    :signature: collection.lastIndexWhere(predicate)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg predicate: function of one argument to apply on every value
    :argType predicate: lambda
    :returnType: integer

    .. code::

        yaql> [1, 2, 3, 2].lastIndexWhere($ = 2)
        3
    """
    index = -1
    for i, t in enumerate(collection):
        if predicate(t):
            index = i
    return index