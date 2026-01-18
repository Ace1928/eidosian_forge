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
def handle_pageBreak(self, slow=None):
    """some might choose not to end all the frames"""
    if self._pageBreakQuick and (not slow):
        self.handle_pageEnd()
    else:
        n = len(self._hanging)
        while len(self._hanging) == n:
            self.handle_frameEnd()