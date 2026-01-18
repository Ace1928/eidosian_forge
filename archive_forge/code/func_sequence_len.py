import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('sequence', yaqltypes.Sequence())
@specs.extension_method
@specs.name('len')
def sequence_len(sequence):
    """:yaql:len

    Returns length of the list.

    :signature: sequence.len()
    :receiverArg sequence: input sequence
    :argType dict: sequence
    :returnType: integer

    .. code::

        yaql> [0, 1, 2].len()
        3
    """
    return len(sequence)