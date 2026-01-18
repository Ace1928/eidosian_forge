from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def prepare_tag_prefix(self, prefix):
    if not prefix:
        raise EmitterError('tag prefix must not be empty')
    chunks = []
    start = end = 0
    if prefix[0] == u'!':
        end = 1
    while end < len(prefix):
        ch = prefix[end]
        if u'0' <= ch <= u'9' or u'A' <= ch <= u'Z' or u'a' <= ch <= u'z' or (ch in u"-;/?!:@&=+$,_.~*'()[]"):
            end += 1
        else:
            if start < end:
                chunks.append(prefix[start:end])
            start = end = end + 1
            data = utf8(ch)
            for ch in data:
                chunks.append(u'%%%02X' % ord(ch))
    if start < end:
        chunks.append(prefix[start:end])
    return ''.join(chunks)