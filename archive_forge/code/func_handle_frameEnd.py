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
def handle_frameEnd(self, resume=0):
    """ Handles the semantics of the end of a frame. This includes the selection of
            the next frame or if this is the last frame then invoke pageEnd.
        """
    self._removeVars(('frame',))
    self._leftExtraIndent = self.frame._leftExtraIndent
    self._rightExtraIndent = self.frame._rightExtraIndent
    self._frameBGs = self.frame._frameBGs
    if hasattr(self, '_nextFrameIndex'):
        self.frame = self.pageTemplate.frames[self._nextFrameIndex]
        self.frame._debug = self._debug
        del self._nextFrameIndex
        self.handle_frameBegin(resume)
    else:
        f = self.frame
        if hasattr(f, 'lastFrame') or f is self.pageTemplate.frames[-1]:
            self.handle_pageEnd()
            self.frame = None
        else:
            self.frame = self.pageTemplate.frames[self.pageTemplate.frames.index(f) + 1]
            self.frame._debug = self._debug
            self.handle_frameBegin()