import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.method
@specs.parameter('collection', yaqltypes.Iterable(validators=[lambda x: not isinstance(x, utils.SetType)]))
@specs.parameter('value', nullable=True)
@specs.parameter('position', int)
@specs.name('insert')
def iter_insert(collection, position, value):
    """:yaql:insert

    Returns collection with inserted value at the given position.

    :signature: collection.insert(position, value)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg position: index for insertion. value is inserted in the end if
        position greater than collection size
    :argType position: integer
    :arg value: value to be inserted
    :argType value: any
    :returnType: iterable

    .. code::

        yaql> [0, 1, 3].insert(2, 2)
        [0, 1, 2, 3]
    """
    i = -1
    for i, t in enumerate(collection):
        if i == position:
            yield value
        yield t
    if position > i:
        yield value