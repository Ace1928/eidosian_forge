from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
def dumpProcessedFrags(P, label='processed_frags'):
    if isinstance(P2.frags[0], list):
        _F = {}
        _S = [].append

        def _showWord(w):
            t = [].append
            for _ in w[1:]:
                fid = id(_[0])
                if fid not in _F:
                    _F[fid] = (len(_F), _[0])
                t('(__frag_%s__, %r)' % (_F[fid][0], _[1]))
            return '  %s([%s, %s]),' % (w.__class__.__name__, w[0], ', '.join(t.__self__))
        for _ in P2.frags:
            _S(_showWord(_))
        print('from reportlab.platypus.paragraph import _HSFrag, _SplitFragHS, _SplitFragHY, _SplitFrag, _getFragWords\nfrom reportlab.platypus.paraparser import ParaFrag\nfrom reportlab.lib.colors import Color')
        print('\n'.join(('__frag_%s__ = %r' % _ for _ in sorted(_F.values()))))
        print('%s=[\n%s  ]' % (processed_frags, '\n'.join(_S.__self__)))
        print('print(_getFragWords(processed_frags))')