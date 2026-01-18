from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def rcCanvLine(xs, ys, xe, ye):
    if xs == xe and (xs >= rrd.x1 or xs <= rrd.x0) or (ys == ye and (ys >= rrd.y1 or ys <= rrd.y0)):
        aSL(RoundingRectLine(xs, ys, xe, ye, weight, color, cap, dash, join))
    else:
        ocanvline(xs, ys, xe, ye)