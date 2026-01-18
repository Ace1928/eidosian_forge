import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('items', yaqltypes.Iterable())
@specs.no_kwargs
def dict__(items, engine):
    """:yaql:dict

    Returns dictionary with keys and values built on items pairs.

    :signature: dict(items)
    :arg items: list of pairs [key, value] for building dictionary
    :argType items: list
    :returnType: dictionary

    .. code::

        yaql> dict([["a", 2], ["b", 4]])
        {"a": 2, "b": 4}
    """
    result = {}
    for t in items:
        it = iter(t)
        key = next(it)
        value = next(it)
        result[key] = value
        utils.limit_memory_usage(engine, (1, result))
    return utils.FrozenDict(result)