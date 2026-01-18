import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def fixup_chunks(chunks):
    """
    This function takes a list of chunks and produces a list of tokens.
    """
    tag_accum = []
    cur_word = None
    result = []
    for chunk in chunks:
        if isinstance(chunk, tuple):
            if chunk[0] == 'img':
                src = chunk[1]
                tag, trailing_whitespace = split_trailing_whitespace(chunk[2])
                cur_word = tag_token('img', src, html_repr=tag, pre_tags=tag_accum, trailing_whitespace=trailing_whitespace)
                tag_accum = []
                result.append(cur_word)
            elif chunk[0] == 'href':
                href = chunk[1]
                cur_word = href_token(href, pre_tags=tag_accum, trailing_whitespace=' ')
                tag_accum = []
                result.append(cur_word)
            continue
        if is_word(chunk):
            chunk, trailing_whitespace = split_trailing_whitespace(chunk)
            cur_word = token(chunk, pre_tags=tag_accum, trailing_whitespace=trailing_whitespace)
            tag_accum = []
            result.append(cur_word)
        elif is_start_tag(chunk):
            tag_accum.append(chunk)
        elif is_end_tag(chunk):
            if tag_accum:
                tag_accum.append(chunk)
            else:
                assert cur_word, 'Weird state, cur_word=%r, result=%r, chunks=%r of %r' % (cur_word, result, chunk, chunks)
                cur_word.post_tags.append(chunk)
        else:
            assert False
    if not result:
        return [token('', pre_tags=tag_accum)]
    else:
        result[-1].post_tags.extend(tag_accum)
    return result