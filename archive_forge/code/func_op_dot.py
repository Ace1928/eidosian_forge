import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
@specs.parameter('receiver', Yaqlized(can_call_methods=True))
@specs.parameter('expr', yaqltypes.YaqlExpression(expressions.Function))
@specs.name('#operator_.')
def op_dot(receiver, expr, context, engine):
    """:yaql:operator .

    Evaluates expression on receiver and returns its result.

    :signature: receiver.expr
    :arg receiver: yaqlized receiver
    :argType receiver: yaqlized object, initialized with
        yaqlize_methods equal to True
    :arg expr: expression to be evaluated
    :argType expr: expression
    :returnType: any (expression return type)
    """
    settings = yaqlization.get_yaqlization_settings(receiver)
    mappings = _remap_name(expr.name, settings)
    _validate_name(expr.name, settings)
    if not isinstance(mappings, str):
        name = mappings[0]
        if len(mappings) > 0:
            arg_mappings = mappings[1]
        else:
            arg_mappings = {}
    else:
        name = mappings
        arg_mappings = {}
    func = getattr(receiver, name)
    args, kwargs = runner.translate_args(False, expr.args, {})
    args = tuple((arg(utils.NO_VALUE, context, engine) for arg in args))
    for key, value in kwargs.items():
        kwargs[arg_mappings.get(key, key)] = value(utils.NO_VALUE, context, engine)
    res = func(*args, **kwargs)
    _auto_yaqlize(res, settings)
    return res