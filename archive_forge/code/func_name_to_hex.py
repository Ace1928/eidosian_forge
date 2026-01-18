from . import constants, normalization, types
def name_to_hex(name: str, spec: str=constants.CSS3) -> str:
    """
    Convert a color name to a normalized hexadecimal color value.

    The color name will be normalized to lower-case before being looked
    up.

    Examples:

    .. doctest::

        >>> name_to_hex("white")
        '#ffffff'
        >>> name_to_hex("navy")
        '#000080'
        >>> name_to_hex("goldenrod")
        '#daa520'
        >>> name_to_hex("goldenrod", spec=HTML4)
        Traceback (most recent call last):
            ...
        ValueError: "goldenrod" is not defined as a named color in html4.

    :param name: The color name to convert.
    :param spec: The specification from which to draw the list of color
       names. Default is :data:`CSS3`.
    :raises ValueError: when the given name has no definition in the given spec.

    """
    if spec not in constants.SUPPORTED_SPECIFICATIONS:
        raise ValueError(constants.SPECIFICATION_ERROR_TEMPLATE.format(spec=spec))
    hex_value = getattr(constants, f'{spec.upper()}_NAMES_TO_HEX').get(name.lower())
    if hex_value is None:
        raise ValueError(f'"{name}" is not defined as a named color in {spec}')
    return hex_value