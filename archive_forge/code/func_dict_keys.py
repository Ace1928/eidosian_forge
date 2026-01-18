import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('keys')
@specs.method
def dict_keys(d):
    """:yaql:keys

    Returns an iterator over the dictionary keys.

    :signature: dict.keys()
    :receiverArg dict: input dictionary
    :argType dict: dictionary
    :returnType: iterator

    .. code::

        yaql> {"a" => 1, "b" => 2}.keys()
        ["a", "b"]
    """
    return d.keys()