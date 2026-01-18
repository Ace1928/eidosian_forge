import typing as t
from . import Markup
def soft_str(s: t.Any) -> str:
    """Convert an object to a string if it isn't already. This preserves
    a :class:`Markup` string rather than converting it back to a basic
    string, so it will still be marked as safe and won't be escaped
    again.

    >>> value = escape("<User 1>")
    >>> value
    Markup('&lt;User 1&gt;')
    >>> escape(str(value))
    Markup('&amp;lt;User 1&amp;gt;')
    >>> escape(soft_str(value))
    Markup('&lt;User 1&gt;')
    """
    if not isinstance(s, str):
        return str(s)
    return s