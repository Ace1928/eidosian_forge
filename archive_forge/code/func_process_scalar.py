from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def process_scalar(self):
    if self.analysis is None:
        self.analysis = self.analyze_scalar(self.event.value)
    if self.style is None:
        self.style = self.choose_scalar_style()
    split = not self.simple_key_context
    if self.sequence_context and (not self.flow_level):
        self.write_indent()
    if self.style == '"':
        self.write_double_quoted(self.analysis.scalar, split)
    elif self.style == "'":
        self.write_single_quoted(self.analysis.scalar, split)
    elif self.style == '>':
        self.write_folded(self.analysis.scalar)
    elif self.style == '|':
        self.write_literal(self.analysis.scalar, self.event.comment)
    else:
        self.write_plain(self.analysis.scalar, split)
    self.analysis = None
    self.style = None
    if self.event.comment:
        self.write_post_comment(self.event)