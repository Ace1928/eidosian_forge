import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
def rgb_hex_str(self, x):
    """Provides the color corresponding to value `x` in the
        form of a string of hexadecimal values "#RRGGBB".
        """
    return '#%02x%02x%02x' % self.rgb_bytes_tuple(x)