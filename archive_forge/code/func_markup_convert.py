from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
def markup_convert(t):
    htmltag = TestTransformStringUsingParseActions.markup_convert_map[t.markup_symbol]
    return '<{0}>{1}</{2}>'.format(htmltag, t.body, htmltag)