import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def compress_merge_back(tokens, tok):
    """ Merge tok into the last element of tokens (modifying the list of
    tokens in-place).  """
    last = tokens[-1]
    if type(last) is not token or type(tok) is not token:
        tokens.append(tok)
    else:
        text = _unicode(last)
        if last.trailing_whitespace:
            text += last.trailing_whitespace
        text += tok
        merged = token(text, pre_tags=last.pre_tags, post_tags=tok.post_tags, trailing_whitespace=tok.trailing_whitespace)
        merged.annotation = last.annotation
        tokens[-1] = merged