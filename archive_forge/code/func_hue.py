from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def hue(stream, **kwargs):
    """Modify the hue and/or the saturation of the input.

    Args:
        h: Specify the hue angle as a number of degrees. It accepts an expression, and defaults to "0".
        s: Specify the saturation in the [-10,10] range. It accepts an expression and defaults to "1".
        H: Specify the hue angle as a number of radians. It accepts an expression, and defaults to "0".
        b: Specify the brightness in the [-10,10] range. It accepts an expression and defaults to "0".

    Official documentation: `hue <https://ffmpeg.org/ffmpeg-filters.html#hue>`__
    """
    return FilterNode(stream, hue.__name__, kwargs=kwargs).stream()