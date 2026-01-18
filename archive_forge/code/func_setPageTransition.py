import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def setPageTransition(self, effectname=None, duration=1, direction=0, dimension='H', motion='I'):
    """PDF allows page transition effects for use when giving
        presentations.  There are six possible effects.  You can
        just guive the effect name, or supply more advanced options
        to refine the way it works.  There are three types of extra
        argument permitted, and here are the allowed values::
        
            direction_arg = [0,90,180,270]
            dimension_arg = ['H', 'V']
            motion_arg = ['I','O'] (start at inside or outside)
        
        This table says which ones take which arguments::

            PageTransitionEffects = {
                'Split': [direction_arg, motion_arg],
                'Blinds': [dimension_arg],
                'Box': [motion_arg],
                'Wipe' : [direction_arg],
                'Dissolve' : [],
                'Glitter':[direction_arg]
                }
        
        Have fun!
        """
    self._pageTransition = {}
    if not effectname:
        return
    if direction in [0, 90, 180, 270]:
        direction_arg = ('Di', '/%d' % direction)
    else:
        raise pdfdoc.PDFError(' directions allowed are 0,90,180,270')
    if dimension in ['H', 'V']:
        dimension_arg = ('Dm', '/' + dimension)
    else:
        raise pdfdoc.PDFError('dimension values allowed are H and V')
    if motion in ['I', 'O']:
        motion_arg = ('M', '/' + motion)
    else:
        raise pdfdoc.PDFError('motion values allowed are I and O')
    PageTransitionEffects = {'Split': [direction_arg, motion_arg], 'Blinds': [dimension_arg], 'Box': [motion_arg], 'Wipe': [direction_arg], 'Dissolve': [], 'Glitter': [direction_arg]}
    try:
        args = PageTransitionEffects[effectname]
    except KeyError:
        raise pdfdoc.PDFError('Unknown Effect Name "%s"' % effectname)
    transDict = {}
    transDict['Type'] = '/Trans'
    transDict['D'] = '%d' % duration
    transDict['S'] = '/' + effectname
    for key, value in args:
        transDict[key] = value
    self._pageTransition = transDict