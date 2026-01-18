import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.method
@specs.parameter('collection', yaqltypes.Iterable())
@specs.parameter('position', int)
@specs.parameter('count', int)
@specs.parameter('values', yaqltypes.Iterable())
def replace_many(collection, position, values, count=1):
    """:yaql:replaceMany

    Returns collection where [position, position+count) elements are replaced
    with values items.

    :signature: collection.replaceMany(position, values, count => 1)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg position: index to start replace
    :argType position: integer
    :arg values: items to replace
    :argType values: iterable
    :arg count: how many elements to replace, 1 by default
    :argType count: integer
    :returnType: iterable

    .. code::

        yaql> [0, 1, 3, 4, 2].replaceMany(2, [100, 200], 2)
        [0, 1, 100, 200, 2]
    """
    yielded = False
    for i, t in enumerate(collection):
        if count >= 0 and position <= i < position + count or (count < 0 and i >= position):
            if not yielded:
                for v in values:
                    yield v
                yielded = True
        else:
            yield t