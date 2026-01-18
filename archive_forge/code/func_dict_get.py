import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('get')
@specs.method
def dict_get(d, key, default=None):
    """:yaql:get

    Returns value of a dictionary by given key or default if there is
    no such key.

    :signature: dict.get(key, default => null)
    :receiverArg dict: input dictionary
    :argType dict: dictionary
    :arg key: key
    :argType key: keyword
    :arg default: default value to be returned if key is missing in dictionary.
        null by default
    :argType default: any
    :returnType: any (appropriate value type)

    .. code::

        yaql> {"a" => 1, "b" => 2}.get("c")
        null
        yaql> {"a" => 1, "b" => 2}.get("c", 3)
        3
    """
    return d.get(key, default)