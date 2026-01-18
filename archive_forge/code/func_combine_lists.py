import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('left', yaqltypes.Iterable())
@specs.parameter('right', yaqltypes.Iterable())
@specs.name('#operator_+')
def combine_lists(left, right, engine):
    """:yaql:operator +

    Returns two iterables concatenated.

    :signature: left + right
    :arg left: left list
    :argType left: iterable
    :arg right: right list
    :argType right: iterable
    :returnType: iterable

    .. code::

        yaql> [1, 2] + [3]
        [1, 2, 3]
    """
    if isinstance(left, tuple) and isinstance(right, tuple):
        utils.limit_memory_usage(engine, (1, left), (1, right))
        return left + right
    elif isinstance(left, frozenset) and isinstance(right, frozenset):
        utils.limit_memory_usage(engine, (1, left), (1, right))
        return left.union(right)
    return yaql.standard_library.queries.concat(left, right)