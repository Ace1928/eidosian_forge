from __future__ import absolute_import
import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
from . import Version
from .Code import CCodeWriter
from .. import Utils
def mark_pos(self, pos, trace=True):
    if pos is not None:
        CCodeWriter.mark_pos(self, pos, trace)
        if self.funcstate and self.funcstate.scope:
            self.scopes[pos[0].filename][pos[1]].add(self.funcstate.scope)
    if self.last_annotated_pos:
        source_desc, line, _ = self.last_annotated_pos
        pos_code = self.code[source_desc.filename]
        pos_code[line] += self.annotation_buffer.getvalue()
    self.annotation_buffer = StringIO()
    self.last_annotated_pos = pos