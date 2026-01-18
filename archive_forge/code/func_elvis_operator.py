import itertools
from yaql.language import contexts
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('expr', yaqltypes.YaqlExpression())
@specs.inject('operator', yaqltypes.Delegate('#operator_.'))
@specs.name('#operator_?.')
def elvis_operator(operator, receiver, expr):
    """:yaql:operator ?.

    Evaluates expr on receiver if receiver isn't null and returns the result.
    If receiver is null returns null.

    :signature: receiver?.expr
    :arg receiver: object to evaluate expression
    :argType receiver: any
    :arg expr: expression
    :argType expr: expression that can be evaluated as a method
    :returnType: expression result or null

    .. code::

        yaql> [0, 1]?.select($+1)
        [1, 2]
        yaql> null?.select($+1)
        null
    """
    if receiver is None:
        return None
    return operator(receiver, expr)