import re
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('pattern', yaqltypes.String())
@specs.parameter('string', yaqltypes.String())
@specs.name('#operator_!~')
def not_matches_operator_string(string, pattern):
    """:yaql:operator !~

    Returns true if left doesn't match right, false otherwise.

    :signature: left !~ right
    :arg left: string to find match in
    :argType left: string
    :arg right: regex pattern
    :argType right: regex object
    :returnType: boolean

    .. code::

        yaql> "acb" !~ regex("a.c")
        true
        yaql> "abc" !~ regex("a.c")
        false
    """
    return re.search(pattern, string) is None