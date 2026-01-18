from . import dot2tex as d2t
from pyparsing import ParseException
import logging
def dot2tex(dotsource, **kwargs):
    """Process dotsource and return LaTeX code

    Conversion options can be specified as keyword options. Example:
        dot2tex(data,format='tikz',crop=True)

    """
    return d2t.convert_graph(dotsource, **kwargs)