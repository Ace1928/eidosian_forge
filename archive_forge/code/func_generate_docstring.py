from __future__ import annotations
import logging # isort:skip
from inspect import Parameter
from ..models import Marker
def generate_docstring(glyphclass, parameters, extra_docs):
    return f' {_docstring_header(glyphclass)}\n\nArgs:\n{_docstring_args(parameters)}\n\nKeyword args:\n{_docstring_kwargs(parameters)}\n\n{_docstring_other()}\n\nIt is also possible to set the color and alpha parameters of extra glyphs for\nselection, nonselection, hover, or muted. To do so, add the relevant prefix to\nany visual parameter. For example, pass ``nonselection_alpha`` to set the line\nand fill alpha for nonselect, or ``hover_fill_alpha`` to set the fill alpha for\nhover. See the :ref:`ug_styling_plots_glyphs` section of the user guide for\nfull details.\n\nReturns:\n    :class:`~bokeh.models.renderers.GlyphRenderer`\n\n{_docstring_extra(extra_docs)}\n'