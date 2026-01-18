from . import constants, normalization, types
def rgb_to_rgb_percent(rgb_triplet: types.IntTuple) -> types.PercentRGB:
    """
    Convert a 3-:class:`tuple` of :class:`int`, suitable for use in an ``rgb()``
    color triplet, to a 3-:class:`tuple` of percentages suitable for use in
    representing that color.

    .. note:: **Floating-point precision**

       This function makes some trade-offs in terms of the accuracy of the final
       representation. For some common integer values, special-case logic is used to
       ensure a precise result (e.g., integer 128 will always convert to ``"50%"``,
       integer 32 will always convert to ``"12.5%"``), but for all other values a
       standard Python :class:`float` is used and rounded to two decimal places, which
       may result in a loss of precision for some values due to the inherent imprecision
       of `IEEE floating-point numbers <https://en.wikipedia.org/wiki/IEEE_754>`_.

    Examples:

    .. doctest::

        >>> rgb_to_rgb_percent((255, 255, 255))
        PercentRGB(red='100%', green='100%', blue='100%')
        >>> rgb_to_rgb_percent((0, 0, 128))
        PercentRGB(red='0%', green='0%', blue='50%')
        >>> rgb_to_rgb_percent((218, 165, 32))
        PercentRGB(red='85.49%', green='64.71%', blue='12.5%')

    :param rgb_triplet: The ``rgb()`` triplet.

    """
    specials = {255: '100%', 128: '50%', 64: '25%', 32: '12.5%', 16: '6.25%', 0: '0%'}
    return types.PercentRGB._make((specials.get(d, f'{d / 255.0 * 100:.02f}%') for d in normalization.normalize_integer_triplet(rgb_triplet)))