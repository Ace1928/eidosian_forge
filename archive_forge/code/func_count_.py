import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('collection', utils.IteratorType)
@specs.name('len')
@specs.extension_method
def count_(collection):
    """:yaql:len

    Returns the size of the collection.

    :signature: collection.len()
    :receiverArg collection: input collection
    :argType collection: iterable
    :returnType: integer

    .. code::

        yaql> [1, 2].len()
        2
    """
    count = 0
    for t in collection:
        count += 1
    return count