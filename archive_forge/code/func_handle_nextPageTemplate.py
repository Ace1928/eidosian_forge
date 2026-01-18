from reportlab.platypus.flowables import *
from reportlab.platypus.flowables import _ContainerSpace
from reportlab.lib.units import inch
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.frames import Frame
from reportlab.rl_config import defaultPageSize, verbose
import reportlab.lib.sequencer
from reportlab.pdfgen import canvas
from reportlab.lib.utils import isSeq, encode_label, decode_label, annotateException, strTypes
import sys
import logging
def handle_nextPageTemplate(self, pt):
    """On endPage change to the page template with name or index pt"""
    if isinstance(pt, strTypes):
        if hasattr(self, '_nextPageTemplateCycle'):
            del self._nextPageTemplateCycle
        for t in self.pageTemplates:
            if t.id == pt:
                self._nextPageTemplateIndex = self.pageTemplates.index(t)
                return
        raise ValueError("can't find template('%s')" % pt)
    elif isinstance(pt, int):
        if hasattr(self, '_nextPageTemplateCycle'):
            del self._nextPageTemplateCycle
        self._nextPageTemplateIndex = pt
    elif isSeq(pt):
        c = PTCycle()
        for ptn in pt:
            found = 0
            if ptn == '*':
                c._restart = len(c)
                continue
            for t in self.pageTemplates:
                if t.id == ptn:
                    c.append(t)
                    found = 1
            if not found:
                raise ValueError('Cannot find page template called %s' % ptn)
        if not c:
            raise ValueError('No valid page templates in cycle')
        elif c._restart > len(c):
            raise ValueError('Invalid cycle restart position')
        self._nextPageTemplateCycle = c
    else:
        raise TypeError('argument pt should be string or integer or list')