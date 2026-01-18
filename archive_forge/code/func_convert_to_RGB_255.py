import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def convert_to_RGB_255(colors):
    """
    Multiplies each element of a triplet by 255

    Each coordinate of the color tuple is rounded to the nearest float and
    then is turned into an integer. If a number is of the form x.5, then
    if x is odd, the number rounds up to (x+1). Otherwise, it rounds down
    to just x. This is the way rounding works in Python 3 and in current
    statistical analysis to avoid rounding bias

    :param (list) rgb_components: grabs the three R, G and B values to be
        returned as computed in the function
    """
    rgb_components = []
    for component in colors:
        rounded_num = decimal.Decimal(str(component * 255.0)).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_EVEN)
        rounded_num = int(rounded_num)
        rgb_components.append(rounded_num)
    return (rgb_components[0], rgb_components[1], rgb_components[2])