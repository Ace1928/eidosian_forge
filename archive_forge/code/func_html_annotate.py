import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def html_annotate(doclist, markup=default_markup):
    """
    doclist should be ordered from oldest to newest, like::

        >>> version1 = 'Hello World'
        >>> version2 = 'Goodbye World'
        >>> print(html_annotate([(version1, 'version 1'),
        ...                      (version2, 'version 2')]))
        <span title="version 2">Goodbye</span> <span title="version 1">World</span>

    The documents must be *fragments* (str/UTF8 or unicode), not
    complete documents

    The markup argument is a function to markup the spans of words.
    This function is called like markup('Hello', 'version 2'), and
    returns HTML.  The first argument is text and never includes any
    markup.  The default uses a span with a title:

        >>> print(default_markup('Some Text', 'by Joe'))
        <span title="by Joe">Some Text</span>
    """
    tokenlist = [tokenize_annotated(doc, version) for doc, version in doclist]
    cur_tokens = tokenlist[0]
    for tokens in tokenlist[1:]:
        html_annotate_merge_annotations(cur_tokens, tokens)
        cur_tokens = tokens
    cur_tokens = compress_tokens(cur_tokens)
    result = markup_serialize_tokens(cur_tokens, markup)
    return ''.join(result).strip()