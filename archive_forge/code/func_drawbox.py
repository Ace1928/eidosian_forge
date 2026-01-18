from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def drawbox(stream, x, y, width, height, color, thickness=None, **kwargs):
    """Draw a colored box on the input image.

    Args:
        x: The expression which specifies the top left corner x coordinate of the box. It defaults to 0.
        y: The expression which specifies the top left corner y coordinate of the box. It defaults to 0.
        width: Specify the width of the box; if 0 interpreted as the input width. It defaults to 0.
        heigth: Specify the height of the box; if 0 interpreted as the input height. It defaults to 0.
        color: Specify the color of the box to write. For the general syntax of this option, check the "Color" section
            in the ffmpeg-utils manual. If the special value invert is used, the box edge color is the same as the
            video with inverted luma.
        thickness: The expression which sets the thickness of the box edge. Default value is 3.
        w: Alias for ``width``.
        h: Alias for ``height``.
        c: Alias for ``color``.
        t: Alias for ``thickness``.

    Official documentation: `drawbox <https://ffmpeg.org/ffmpeg-filters.html#drawbox>`__
    """
    if thickness:
        kwargs['t'] = thickness
    return FilterNode(stream, drawbox.__name__, args=[x, y, width, height, color], kwargs=kwargs).stream()