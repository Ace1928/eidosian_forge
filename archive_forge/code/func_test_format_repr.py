import pytest
from wasabi.util import color, diff_strings, format_repr, locale_escape, wrap
def test_format_repr():
    obj = {'hello': 'world', 'test': 123}
    formatted = format_repr(obj)
    assert formatted.replace("u'", "'") in ["{'hello': 'world', 'test': 123}", "{'test': 123, 'hello': 'world'}"]
    formatted = format_repr(obj, max_len=10)
    assert formatted.replace("u'", "'") in ["{'hel ...  123}", "{'tes ... rld'}", "{'te ... rld'}"]
    formatted = format_repr(obj, max_len=10, ellipsis='[...]')
    assert formatted.replace("u'", "'") in ["{'hel [...]  123}", "{'tes [...] rld'}", "{'te [...] rld'}"]