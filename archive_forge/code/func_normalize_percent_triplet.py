from . import constants, types
def normalize_percent_triplet(rgb_triplet: types.PercentTuple) -> types.PercentRGB:
    """
    Normalize a percentage ``rgb()`` triplet so that all values are
    within the range 0%..100%.

    Examples:

    .. doctest::

       >>> normalize_percent_triplet(("50%", "50%", "50%"))
       PercentRGB(red='50%', green='50%', blue='50%')
       >>> normalize_percent_triplet(("0%", "100%", "0%"))
       PercentRGB(red='0%', green='100%', blue='0%')
       >>> normalize_percent_triplet(("-10%", "-0%", "500%"))
       PercentRGB(red='0%', green='0%', blue='100%')

    :param rgb_triplet: The percentage `rgb()` triplet to normalize.

    """
    return types.PercentRGB._make((_normalize_percent_rgb(value) for value in rgb_triplet))