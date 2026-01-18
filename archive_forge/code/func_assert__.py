import itertools
from yaql.language import contexts
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.method
@specs.parameter('condition', yaqltypes.Lambda())
@specs.parameter('message', yaqltypes.String())
def assert__(engine, obj, condition, message=u'Assertion failed'):
    """:yaql:assert

    Evaluates condition against object. If it evaluates to true returns the
    object, otherwise throws an exception with provided message.

    :signature: obj.assert(condition, message => "Assertion failed")
    :arg obj: object to evaluate condition on
    :argType obj: any
    :arg condition: lambda function to be evaluated on obj. If result of
        function evaluates to false then trows exception message
    :argType condition: lambda
    :arg message: message to trow if condition returns false
    :argType message: string
    :returnType: obj type or message

    .. code::

        yaql> 12.assert($ < 2)
        Execution exception: Assertion failed
        yaql> 12.assert($ < 20)
        12
        yaql> [].assert($, "Failed assertion")
        Execution exception: Failed assertion
    """
    if utils.is_iterator(obj):
        obj = utils.memorize(obj, engine)
    if not condition(obj):
        raise AssertionError(message)
    return obj