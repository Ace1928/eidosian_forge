from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.MultiLine)
def multi_line(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
.. note::
    For this glyph, the data is not simply an array of scalars, it is an
    "array of arrays".

Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        p = figure(width=300, height=300)
        p.multi_line(xs=[[1, 2, 3], [2, 3, 4]], ys=[[6, 7, 2], [4, 5, 7]],
                    color=['red','green'])

        show(p)

"""