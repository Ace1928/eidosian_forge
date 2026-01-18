from __future__ import annotations
import sys
from . import EpsImagePlugin
def begin_document(self, id=None):
    """Set up printing of a document. (Write PostScript DSC header.)"""
    self.fp.write(b'%!PS-Adobe-3.0\nsave\n/showpage { } def\n%%EndComments\n%%BeginDocument\n')
    self.fp.write(EDROFF_PS)
    self.fp.write(VDI_PS)
    self.fp.write(b'%%EndProlog\n')
    self.isofont = {}