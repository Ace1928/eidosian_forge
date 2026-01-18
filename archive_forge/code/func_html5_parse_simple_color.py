import string
from . import constants, types
def html5_parse_simple_color(value: str) -> types.HTML5SimpleColor:
    """
    Apply the HTML5 simple color parsing algorithm.

    Examples:

    .. doctest::

        >>> html5_parse_simple_color("#ffffff")
        HTML5SimpleColor(red=255, green=255, blue=255)
        >>> html5_parse_simple_color("#fff")
        Traceback (most recent call last):
            ...
        ValueError: An HTML5 simple color must be a string seven characters long.

    :param value: The color to parse.
    :type value: :class:`str`, which must consist of exactly
        the character ``"#"`` followed by six hexadecimal digits
    :raises ValueError: when the given value is not a Unicode string of
       length 7, consisting of exactly the character ``#`` followed by
       six hexadecimal digits.


    """
    if not isinstance(value, str) or len(value) != 7:
        raise ValueError('An HTML5 simple color must be a Unicode string seven characters long.')
    if not value.startswith('#'):
        raise ValueError("An HTML5 simple color must begin with the character '#' (U+0023).")
    if not all((c in string.hexdigits for c in value[1:])):
        raise ValueError('An HTML5 simple color must contain exactly six ASCII hex digits.')
    return types.HTML5SimpleColor(int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))