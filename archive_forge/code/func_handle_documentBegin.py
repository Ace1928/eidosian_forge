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
def handle_documentBegin(self):
    """implement actions at beginning of document"""
    self._hanging = [PageBegin]
    if isinstance(self._firstPageTemplateIndex, list):
        self.handle_nextPageTemplate(self._firstPageTemplateIndex)
        self._setPageTemplate()
    else:
        self.pageTemplate = self.pageTemplates[self._firstPageTemplateIndex]
    self.page = 0
    self.beforeDocument()