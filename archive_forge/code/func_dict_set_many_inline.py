import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.no_kwargs
@specs.method
@specs.parameter('args', utils.MappingRule)
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('set')
def dict_set_many_inline(engine, d, *args):
    """:yaql:set

    Returns dict with args keys set to args values.

    :signature: dict.set([args])
    :receiverArg dict: input dictionary
    :argType dict: dictionary
    :arg [args]: key-values to be set on input dict
    :argType [args]: chain of mappings
    :returnType: dictionary

    .. code::

        yaql> {"a" => 1, "b" => 2}.set("b" => 3, "c" => 4)
        {"a": 1, "c": 4, "b": 3}
    """
    utils.limit_memory_usage(engine, (1, d), *((1, arg) for arg in args))
    return utils.FrozenDict(itertools.chain(d.items(), ((t.source, t.destination) for t in args)))