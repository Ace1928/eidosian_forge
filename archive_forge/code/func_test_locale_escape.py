import pytest
from wasabi.util import color, diff_strings, format_repr, locale_escape, wrap
@pytest.mark.parametrize('text,non_ascii', [('abc', ['abc']), ('✔ abc', ['? abc']), ('👻', ['??', '?'])])
def test_locale_escape(text, non_ascii):
    result = locale_escape(text)
    assert result == text or result in non_ascii
    print(result)