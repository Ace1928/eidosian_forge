import itertools
from yaql.language import contexts
from yaql.language import expressions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('receiver', contexts.ContextBase)
@specs.parameter('expr', yaqltypes.Lambda(with_context=True, method=True))
@specs.name('#operator_.')
def op_dot_context(receiver, expr):
    return expr(receiver['$0'], receiver)