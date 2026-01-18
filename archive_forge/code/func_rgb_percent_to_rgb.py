from . import constants, normalization, types
def rgb_percent_to_rgb(rgb_percent_triplet: types.PercentTuple) -> types.IntegerRGB:
    """

    Convert a 3-:class:`tuple` of percentages, suitable for use in an ``rgb()``
    color triplet, to a 3-:class:`tuple` of :class:`int` suitable for use in
    representing that color.

    Some precision may be lost in this conversion. See the note
    regarding precision for :func:`~webcolors.rgb_to_rgb_percent` for
    details.

    Examples:

    .. doctest::

        >>> rgb_percent_to_rgb(("100%", "100%", "100%"))
        IntegerRGB(red=255, green=255, blue=255)
        >>> rgb_percent_to_rgb(("0%", "0%", "50%"))
        IntegerRGB(red=0, green=0, blue=128)
        >>> rgb_percent_to_rgb(("85.49%", "64.71%", "12.5%"))
        IntegerRGB(red=218, green=165, blue=32)

    :param rgb_percent_triplet: The ``rgb()`` triplet.

    """
    return types.IntegerRGB._make(map(normalization._percent_to_integer, normalization.normalize_percent_triplet(rgb_percent_triplet)))