from . import constants, normalization, types
def hex_to_rgb_percent(hex_value: str) -> types.PercentRGB:
    """
    Convert a hexadecimal color value to a 3-:class:`tuple` of percentages
    suitable for use in an ``rgb()`` triplet representing that color.

    The hexadecimal value will be normalized before being converted.

    Examples:

    .. doctest::

        >>> hex_to_rgb_percent("#ffffff")
        PercentRGB(red='100%', green='100%', blue='100%')
        >>> hex_to_rgb_percent("#000080")
        PercentRGB(red='0%', green='0%', blue='50%')

    :param hex_value: The hexadecimal color value to convert.
    :raises ValueError: when the supplied hex value is invalid.

    """
    return rgb_to_rgb_percent(hex_to_rgb(hex_value))