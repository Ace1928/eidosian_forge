import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('set')
@specs.method
@specs.no_kwargs
def dict_set(engine, d, key, value):
    """:yaql:set

    Returns dict with provided key set to value.

    :signature: dict.set(key, value)
    :receiverArg dict: input dictionary
    :argType dict: dictionary
    :arg key: key
    :argType key: keyword
    :arg value: value to be set to input key
    :argType value: any
    :returnType: dictionary

    .. code::

        yaql> {"a" => 1, "b" => 2}.set("c", 3)
        {"a": 1, "b": 2, "c": 3}
        yaql> {"a" => 1, "b" => 2}.set("b", 3)
        {"a": 1, "b": 3}
    """
    utils.limit_memory_usage(engine, (1, d), (1, key), (1, value))
    return utils.FrozenDict(itertools.chain(d.items(), ((key, value),)))