from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('args', yaqltypes.Lambda())
def select_case(*args):
    """:yaql:selectCase

    Returns a zero-based index of the first predicate evaluated to true. If
    there is no such predicate, returns the count of arguments. All the
    predicates after the first one which was evaluated to true remain
    unevaluated.

    :signature: selectCase([args])
    :arg [args]: predicates to check for true
    :argType [args]: chain of predicates
    :returnType: integer

    .. code::

        yaql> selectCase("ab" > "abc", "ab" >= "abc", "ab" < "abc")
        2
    """
    index = 0
    for f in args:
        if f():
            return index
        index += 1
    return index