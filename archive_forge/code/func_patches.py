from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.Patches)
def patches(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
.. note::
    For this glyph, the data is not simply an array of scalars, it is an
    "array of arrays".

Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        p = figure(width=300, height=300)
        p.patches(xs=[[1,2,3],[4,5,6,5]], ys=[[1,2,1],[4,5,5,4]],
                  color=["#43a2ca", "#a8ddb5"])

        show(p)

"""