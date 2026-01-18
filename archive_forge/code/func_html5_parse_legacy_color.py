import string
from . import constants, types
def html5_parse_legacy_color(value: str) -> types.HTML5SimpleColor:
    """
    Apply the HTML5 legacy color parsing algorithm.

    Note that, since this algorithm is intended to handle many types of
    malformed color values present in real-world Web documents, it is
    *extremely* forgiving of input, but the results of parsing inputs
    with high levels of "junk" (i.e., text other than a color value)
    may be surprising.

    Examples:

    .. doctest::

        >>> html5_parse_legacy_color("black")
        HTML5SimpleColor(red=0, green=0, blue=0)
        >>> html5_parse_legacy_color("chucknorris")
        HTML5SimpleColor(red=192, green=0, blue=0)
        >>> html5_parse_legacy_color("Window")
        HTML5SimpleColor(red=0, green=13, blue=0)

    :param value: The color to parse.

    :raises ValueError: when the given value is not a Unicode string, when it is the
       empty string, or when it is precisely the string ``"transparent"``.

    """
    if not isinstance(value, str):
        raise ValueError('HTML5 legacy color parsing requires a Unicode string as input.')
    if value == '':
        raise ValueError('HTML5 legacy color parsing forbids empty string as a value.')
    value = value.strip()
    if value.lower() == 'transparent':
        raise ValueError('HTML5 legacy color parsing forbids "transparent" as a value.')
    keyword_hex = constants.CSS3_NAMES_TO_HEX.get(value.lower())
    if keyword_hex is not None:
        return html5_parse_simple_color(keyword_hex)
    if len(value) == 4 and value.startswith('#') and all((c in string.hexdigits for c in value[1:])):
        result = types.HTML5SimpleColor(int(value[1], 16) * 17, int(value[2], 16) * 17, int(value[3], 16) * 17)
        return result
    value = ''.join(('00' if ord(c) > 65535 else c for c in value))
    if len(value) > 128:
        value = value[:128]
    if value.startswith('#'):
        value = value[1:]
    value = ''.join((c if c in string.hexdigits else '0' for c in value))
    while len(value) == 0 or len(value) % 3 != 0:
        value += '0'
    length = int(len(value) / 3)
    red = value[:length]
    green = value[length:length * 2]
    blue = value[length * 2:]
    if length > 8:
        red, green, blue = (red[length - 8:], green[length - 8:], blue[length - 8:])
        length = 8
    while length > 2 and (red[0] == '0' and green[0] == '0' and (blue[0] == '0')):
        red, green, blue = (red[1:], green[1:], blue[1:])
        length -= 1
    if length > 2:
        red, green, blue = (red[:2], green[:2], blue[:2])
    return types.HTML5SimpleColor(int(red, 16), int(green, 16), int(blue, 16))