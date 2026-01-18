import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
def asString(self, format, verbose=None, preview=0, **kw):
    """Converts to an 8 bit string in given format."""
    assert format in self._saveModes, 'Unknown file format "%s"' % format
    from reportlab import rl_config
    if format == 'pdf':
        from reportlab.graphics import renderPDF
        return renderPDF.drawToString(self)
    elif format in self._bmModes:
        from reportlab.graphics import renderPM
        return renderPM.drawToString(self, fmt=format, showBoundary=getattr(self, 'showBorder', rl_config.showBoundary), **_extraKW(self, '_renderPM_', **kw))
    elif format == 'eps':
        try:
            from rlextra.graphics import renderPS_SEP as renderPS
        except ImportError:
            from reportlab.graphics import renderPS
        return renderPS.drawToString(self, preview=preview, showBoundary=getattr(self, 'showBorder', rl_config.showBoundary))
    elif format == 'ps':
        from reportlab.graphics import renderPS
        return renderPS.drawToString(self, showBoundary=getattr(self, 'showBorder', rl_config.showBoundary))
    elif format == 'py':
        return self._renderPy()
    elif format == 'svg':
        from reportlab.graphics import renderSVG
        return renderSVG.drawToString(self, showBoundary=getattr(self, 'showBorder', rl_config.showBoundary), **_extraKW(self, '_renderSVG_', **kw))