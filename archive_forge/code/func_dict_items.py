import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('items')
@specs.method
def dict_items(d):
    """:yaql:items

    Returns an iterator over pairs [key, value] of input dict.

    :signature: dict.items()
    :receiverArg dict: input dictionary
    :argType dict: dictionary
    :returnType: iterator

    .. code::

        yaql> {"a" => 1, "b" => 2}.items()
        [["a", 1], ["b", 2]]
    """
    return d.items()