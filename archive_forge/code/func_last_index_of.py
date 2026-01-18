import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.method
@specs.parameter('collection', yaqltypes.Iterable())
def last_index_of(collection, item):
    """:yaql:lastIndexOf

    Returns the index in the collection of the last item which value is item.
    -1 is a return value if there is no such item

    :signature: collection.lastIndexOf(item)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg item: value to find in collection
    :argType item: any
    :returnType: integer

    .. code::

        yaql> [1, 2, 3, 2].lastIndexOf(2)
        3
    """
    index = -1
    for i, t in enumerate(collection):
        if t == item:
            index = i
    return index