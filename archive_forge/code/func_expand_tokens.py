import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def expand_tokens(tokens, equal=False):
    """Given a list of tokens, return a generator of the chunks of
    text for the data in the tokens.
    """
    for token in tokens:
        yield from token.pre_tags
        if not equal or not token.hide_when_equal:
            if token.trailing_whitespace:
                yield (token.html() + token.trailing_whitespace)
            else:
                yield token.html()
        yield from token.post_tags