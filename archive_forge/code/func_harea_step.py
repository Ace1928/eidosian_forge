from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.HAreaStep)
def harea_step(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        plot = figure(width=300, height=300)
        plot.harea_step(x1=[1, 2, 3], x2=[0, 0, 0], y=[1, 4, 2],
                        step_mode="after", fill_color="#99D594")

        show(plot)

"""