import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('left', utils.MappingType)
@specs.parameter('right', utils.MappingType)
@specs.name('#operator_+')
def combine_dicts(left, right, engine):
    """:yaql:operator +

    Returns combined left and right dictionaries.

    :signature: left + right
    :arg left: left dictionary
    :argType left: mapping
    :arg right: right dictionary
    :argType right: mapping
    :returnType: mapping

    .. code::

        yaql> {"a" => 1, b => 2} + {"b" => 3, "c" => 4}
        {"a": 1, "c": 4, "b": 3}
    """
    utils.limit_memory_usage(engine, (1, left), (1, right))
    d = dict(left)
    d.update(right)
    return utils.FrozenDict(d)