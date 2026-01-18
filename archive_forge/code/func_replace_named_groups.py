import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc
from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
def replace_named_groups(pattern):
    """
    Find named groups in `pattern` and replace them with the group name. E.g.,
    1. ^(?P<a>\\w+)/b/(\\w+)$ ==> ^<a>/b/(\\w+)$
    2. ^(?P<a>\\w+)/b/(?P<c>\\w+)/$ ==> ^<a>/b/<c>/$
    3. ^(?P<a>\\w+)/b/(\\w+) ==> ^<a>/b/(\\w+)
    4. ^(?P<a>\\w+)/b/(?P<c>\\w+) ==> ^<a>/b/<c>
    """
    group_pattern_and_name = [(pattern[start:end], match[1]) for start, end, match in _find_groups(pattern, named_group_matcher)]
    for group_pattern, group_name in group_pattern_and_name:
        pattern = pattern.replace(group_pattern, group_name)
    return pattern