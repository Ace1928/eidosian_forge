from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def prepare_tag(self, tag):
    if not tag:
        raise EmitterError('tag must not be empty')
    if tag == u'!':
        return tag
    handle = None
    suffix = tag
    prefixes = sorted(self.tag_prefixes.keys())
    for prefix in prefixes:
        if tag.startswith(prefix) and (prefix == u'!' or len(prefix) < len(tag)):
            handle = self.tag_prefixes[prefix]
            suffix = tag[len(prefix):]
    chunks = []
    start = end = 0
    while end < len(suffix):
        ch = suffix[end]
        if u'0' <= ch <= u'9' or u'A' <= ch <= u'Z' or u'a' <= ch <= u'z' or (ch in u"-;/?:@&=+$,_.~*'()[]") or (ch == u'!' and handle != u'!'):
            end += 1
        else:
            if start < end:
                chunks.append(suffix[start:end])
            start = end = end + 1
            data = utf8(ch)
            for ch in data:
                chunks.append(u'%%%02X' % ord(ch))
    if start < end:
        chunks.append(suffix[start:end])
    suffix_text = ''.join(chunks)
    if handle:
        return u'%s%s' % (handle, suffix_text)
    else:
        return u'!<%s>' % suffix_text