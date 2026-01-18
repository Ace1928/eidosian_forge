from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.HexTile)
def hex_tile(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        plot = figure(width=300, height=300, match_aspect=True)
        plot.hex_tile(r=[0, 0, 1], q=[1, 2, 2], fill_color="#74ADD1")

        show(plot)

"""