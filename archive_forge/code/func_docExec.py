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
def docExec(self, stmt, lifetime):
    stmt = stmt.strip()
    NS = self._nameSpace
    K0 = list(NS.keys())
    try:
        if lifetime not in self._allowedLifetimes:
            raise ValueError('bad lifetime %r not in %r' % (lifetime, self._allowedLifetimes))
        exec(stmt, NS)
    except:
        K1 = [k for k in NS if k not in K0]
        for k in K1:
            del NS[k]
        annotateException('\ndocExec %s lifetime=%r failed!\n' % (stmt, lifetime))
    self._addVars([k for k in NS.keys() if k not in K0], lifetime)