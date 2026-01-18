import re
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('pattern', yaqltypes.String())
@specs.parameter('string', yaqltypes.String())
@specs.name('#operator_=~')
def matches_operator_string(string, pattern):
    """:yaql:operator =~

    Returns true if left matches right, false otherwise.

    :signature: left =~ right
    :arg left: string to find match in
    :argType left: string
    :arg right: regex pattern
    :argType right: string
    :returnType: boolean

    .. code::

        yaql> "abc" =~ "a.c"
        true
    """
    return re.search(pattern, string) is not None