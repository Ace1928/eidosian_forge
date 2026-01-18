import itertools
from yaql.language import contexts
from yaql.language import expressions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.parameter('key', yaqltypes.Keyword())
@specs.name('#operator_.')
def dict_keyword_access(d, key):
    """:yaql:operator .

    Returns dict's key value.

    :signature: left.right
    :arg left: input dictionary
    :argType left: mapping
    :arg right: key
    :argType right: keyword
    :returnType: any (appropriate value type)

    .. code::

        yaql> {a => 2, b => 2}.a
        2
    """
    return d.get(key)