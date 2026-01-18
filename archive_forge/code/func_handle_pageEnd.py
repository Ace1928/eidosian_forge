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
def handle_pageEnd(self):
    """ show the current page
            check the next page template
            hang a page begin
        """
    self._removeVars(('page', 'frame'))
    self._leftExtraIndent = self.frame._leftExtraIndent
    self._rightExtraIndent = self.frame._rightExtraIndent
    self._frameBGs = self.frame._frameBGs
    if self._curPageFlowableCount == 0:
        self._emptyPages += 1
    else:
        self._emptyPages = 0
    if self._emptyPages >= self._emptyPagesAllowed:
        if 1:
            ident = 'More than %d pages generated without content - halting layout.  Likely that a flowable is too large for any frame.' % self._emptyPagesAllowed
            raise LayoutError(ident)
        else:
            pass
    else:
        if self._onProgress:
            self._onProgress('PAGE', self.canv.getPageNumber())
        self.pageTemplate.afterDrawPage(self.canv, self)
        self.pageTemplate.onPageEnd(self.canv, self)
        self.afterPage()
        if self._debug:
            logger.debug('ending page %d' % self.page)
        self.canv.setPageRotation(getattr(self.pageTemplate, 'rotation', self.rotation))
        self.canv.showPage()
        self._setPageTemplate()
        if self._emptyPages == 0:
            pass
    self._hanging.append(PageBegin)