import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
@_docstring.dedent_interpd
def set_text_props(self, **kwargs):
    """
        Update the text properties.

        Valid keyword arguments are:

        %(Text:kwdoc)s
        """
    self._text._internal_update(kwargs)
    self.stale = True