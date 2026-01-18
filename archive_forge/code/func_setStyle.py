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
def setStyle(self, tblstyle):
    if not isinstance(tblstyle, TableStyle):
        tblstyle = TableStyle(tblstyle)
    for cmd in tblstyle.getCommands():
        if len(cmd) >= 3:
            c, (sc, sr), (ec, er) = cmd[0:3]
            if isinstance(sc, str) or isinstance(ec, str) or (isinstance(sr, str) and sr not in _SPECIALROWS) or (isinstance(er, str) and er not in _SPECIALROWS):
                raise ValueError(f'bad style command {cmd!r} illegal of invalid string coordinate\nonly rows may be strings with values in {_SPECIALROWS!r}')
        self._addCommand(cmd)
    for k, v in tblstyle._opts.items():
        setattr(self, k, v)
    for a in ('spaceBefore', 'spaceAfter'):
        if not hasattr(self, a) and hasattr(tblstyle, a):
            setattr(self, a, getattr(tblstyle, a))