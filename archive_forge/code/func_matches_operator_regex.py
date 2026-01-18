import re
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('regexp', REGEX_TYPE)
@specs.parameter('string', yaqltypes.String())
@specs.name('#operator_=~')
def matches_operator_regex(string, regexp):
    """:yaql:operator =~

    Returns true if left matches right, false otherwise.

    :signature: left =~ right
    :arg left: string to find match in
    :argType left: string
    :arg right: regex pattern
    :argType right: regex
    :returnType: boolean

    .. code::

        yaql> "abc" =~ regex("a.c")
        true
    """
    return regexp.search(string) is not None