from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def prepare_anchor(self, anchor):
    if not anchor:
        raise EmitterError('anchor must not be empty')
    for ch in anchor:
        if not check_anchorname_char(ch):
            raise EmitterError('invalid character %r in the anchor: %r' % (utf8(ch), utf8(anchor)))
    return anchor