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
def handle_keepWithNext(self, flowables):
    """implements keepWithNext"""
    i = 0
    n = len(flowables)
    while i < n and flowables[i].getKeepWithNext() and _ktAllow(flowables[i]):
        i += 1
    if i:
        if i < n and _ktAllow(flowables[i]):
            i += 1
        K = self.keepTogetherClass(flowables[:i])
        mbe = getattr(self, '_multiBuildEdits', None)
        if mbe:
            for f in K._content[:-1]:
                if hasattr(f, 'keepWithNext'):
                    mbe((setattr, f, 'keepWithNext', f.keepWithNext))
                else:
                    mbe((delattr, f, 'keepWithNext'))
                f.__dict__['keepWithNext'] = 0
        else:
            for f in K._content[:-1]:
                f.__dict__['keepWithNext'] = 0
        del flowables[:i]
        flowables.insert(0, K)