from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.compat import utf8, unichr, PY3, check_anchorname_char, nprint  # NOQA
def reset_scanner(self):
    self.done = False
    self.flow_context = []
    self.tokens = []
    self.fetch_stream_start()
    self.tokens_taken = 0
    self.indent = -1
    self.indents = []
    self.allow_simple_key = True
    self.possible_simple_keys = {}