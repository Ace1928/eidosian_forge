from . import constants, normalization, types
def hex_to_name(hex_value: str, spec: str=constants.CSS3) -> str:
    """
    Convert a hexadecimal color value to its corresponding normalized
    color name, if any such name exists.

    The hexadecimal value will be normalized before being looked up.

    .. note:: **Spelling variants**

       Some values representing named gray colors can map to either of two names in
       CSS3, because it supports both ``"gray"`` and ``"grey"`` spelling variants for
       those colors. This function will always return the variant spelled ``"gray"``
       (such as ``"lightgray"`` instead of ``"lightgrey"``). See :ref:`the documentation
       on name conventions <color-name-conventions>` for details.

    Examples:

    .. doctest::

        >>> hex_to_name("#ffffff")
        'white'
        >>> hex_to_name("#fff")
        'white'
        >>> hex_to_name("#000080")
        'navy'
        >>> hex_to_name("#daa520")
        'goldenrod'
        >>> hex_to_name("#daa520", spec=HTML4)
        Traceback (most recent call last):
            ...
        ValueError: "#daa520" has no defined color name in html4.

    :param hex_value: The hexadecimal color value to convert.
    :param spec: The specification from which to draw the list of color
       names. Default is :data:`CSS3`.
    :raises ValueError: when the given color has no name in the given
       spec, or when the supplied hex value is invalid.

    """
    if spec not in constants.SUPPORTED_SPECIFICATIONS:
        raise ValueError(constants.SPECIFICATION_ERROR_TEMPLATE.format(spec=spec))
    name = getattr(constants, f'{spec.upper()}_HEX_TO_NAMES').get(normalization.normalize_hex(hex_value))
    if name is None:
        raise ValueError(f'"{hex_value}" has no defined color name in {spec}.')
    return name