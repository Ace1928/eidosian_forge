import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
def p_unary(this, p):
    if p[1] in yaql_operators.operators:
        alias = this._aliases.get(p.slice[1].type)
        p[0] = expressions.UnaryOperator(p[1], p[2], alias)
    else:
        alias = this._aliases.get(p.slice[2].type)
        p[0] = expressions.UnaryOperator(p[2], p[1], alias)