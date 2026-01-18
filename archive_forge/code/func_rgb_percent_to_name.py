from . import constants, normalization, types
def rgb_percent_to_name(rgb_percent_triplet: types.PercentTuple, spec: str=constants.CSS3) -> str:
    """
    Convert a 3-:class:`tuple` of percentages, suitable for use in an ``rgb()``
    color triplet, to its corresponding normalized color name, if any
    such name exists.

    To determine the name, the triplet will be converted to a
    normalized hexadecimal value.

    .. note:: **Spelling variants**

       Some values representing named gray colors can map to either of two names in
       CSS3, because it supports both ``"gray"`` and ``"grey"`` spelling variants for
       those colors. This function will always return the variant spelled ``"gray"``
       (such as ``"lightgray"`` instead of ``"lightgrey"``). See :ref:`the documentation
       on name conventions <color-name-conventions>` for details.

    Examples:

    .. doctest::

        >>> rgb_percent_to_name(("100%", "100%", "100%"))
        'white'
        >>> rgb_percent_to_name(("0%", "0%", "50%"))
        'navy'
        >>> rgb_percent_to_name(("85.49%", "64.71%", "12.5%"))
        'goldenrod'

    :param rgb_percent_triplet: The ``rgb()`` triplet.
    :param spec: The specification from which to draw the list of color
        names. Default is :data:`CSS3`.
    :raises ValueError: when the given color has no name in the given spec.

    """
    return rgb_to_name(rgb_percent_to_rgb(normalization.normalize_percent_triplet(rgb_percent_triplet)), spec=spec)