import string
from . import constants, types
def html5_serialize_simple_color(simple_color: types.IntTuple) -> str:
    """
    Apply the HTML5 simple color serialization algorithm.

    Examples:

    .. doctest::

        >>> html5_serialize_simple_color((0, 0, 0))
        '#000000'
        >>> html5_serialize_simple_color((255, 255, 255))
        '#ffffff'

    :param simple_color: The color to serialize.

    """
    red, green, blue = simple_color
    result = '#'
    format_string = '{:02x}'
    result += format_string.format(red)
    result += format_string.format(green)
    result += format_string.format(blue)
    return result