import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.method
@specs.parameter('collection', yaqltypes.Iterable())
def index_of(collection, item):
    """:yaql:indexOf

    Returns the index in the collection of the first item which value is item.
    -1 is a return value if there is no such item

    :signature: collection.indexOf(item)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg item: value to find in collection
    :argType item: any
    :returnType: integer

    .. code::

        yaql> [1, 2, 3, 2].indexOf(2)
        1
        yaql> [1, 2, 3, 2].indexOf(102)
        -1
    """
    for i, t in enumerate(collection):
        if t == item:
            return i
    return -1