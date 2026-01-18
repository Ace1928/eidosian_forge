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
def docAssign(self, var, expr, lifetime):
    if not isinstance(expr, strTypes):
        expr = str(expr)
    expr = expr.strip()
    var = var.strip()
    self.docExec('%s=(%s)' % (var.strip(), expr.strip()), lifetime)