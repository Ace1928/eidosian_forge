from . import constants, normalization, types
def name_to_rgb(name: str, spec: str=constants.CSS3) -> types.IntegerRGB:
    """
    Convert a color name to a 3-:class:`tuple` of :class:`int` suitable for use in
    an ``rgb()`` triplet specifying that color.

    The color name will be normalized to lower-case before being looked
    up.

    Examples:

    .. doctest::

        >>> name_to_rgb("white")
        IntegerRGB(red=255, green=255, blue=255)
        >>> name_to_rgb("navy")
        IntegerRGB(red=0, green=0, blue=128)
        >>> name_to_rgb("goldenrod")
        IntegerRGB(red=218, green=165, blue=32)

    :param name: The color name to convert.
    :param spec: The specification from which to draw the list of color
       names. Default is :data:`CSS3.`
    :raises ValueError: when the given name has no definition in the given spec.

    """
    return hex_to_rgb(name_to_hex(name, spec=spec))