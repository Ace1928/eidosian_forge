import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('lst', yaqltypes.Sequence(), alias='list')
@specs.parameter('index', int, nullable=False)
@specs.name('#indexer')
def list_indexer(lst, index):
    """:yaql:operator indexer

    Returns value of sequence by given index.

    :signature: left[right]
    :arg left: input sequence
    :argType left: sequence
    :arg right: index
    :argType right: integer
    :returnType: any (appropriate value type)

    .. code::

        yaql> ["a", "b"][0]
        "a"
    """
    return lst[index]