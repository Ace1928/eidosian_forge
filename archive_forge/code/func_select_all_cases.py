from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('args', yaqltypes.Lambda())
def select_all_cases(*args):
    """:yaql:selectAllCases

    Evaluates input predicates and returns an iterator to collection of
    zero-based indexes of predicates which were evaluated to true. The actual
    evaluation is done lazily as the iterator advances, not during the
    function call.

    :signature: selectAllCases([args])
    :arg [args]: predicates to check for true
    :argType [args]: chain of predicates
    :returnType: iterator

    .. code::

        yaql> selectAllCases("ab" > "abc", "ab" <= "abc", "ab" < "abc")
        [1, 2]
    """
    for i, f in enumerate(args):
        if f():
            yield i