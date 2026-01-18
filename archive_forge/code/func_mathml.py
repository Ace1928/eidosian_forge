from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.MathMLGlyph)
def mathml(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, show

        p = figure(width=300, height=300)
        p.mathml(x=[0], y=[0], text=['''
          <math display="block">
            <mrow>
              <msup>
                <mi>x</mi>
                <mn>2</mn>
              </msup>
              <msup>
                <mi>y</mi>
                <mn>2</mn>
              </msup>
            </mrow>
          </math>
        '''])

        show(p)

"""