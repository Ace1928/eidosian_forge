from itertools import chain
import re
import warnings
from xml.sax.saxutils import unescape
from bleach import html5lib_shim
from bleach import parse_shim
def sanitize_characters(self, token):
    """Handles Characters tokens

        Our overridden tokenizer doesn't do anything with entities. However,
        that means that the serializer will convert all ``&`` in Characters
        tokens to ``&amp;``.

        Since we don't want that, we extract entities here and convert them to
        Entity tokens so the serializer will let them be.

        :arg token: the Characters token to work on

        :returns: a list of tokens

        """
    data = token.get('data', '')
    if not data:
        return token
    data = INVISIBLE_CHARACTERS_RE.sub(INVISIBLE_REPLACEMENT_CHAR, data)
    token['data'] = data
    if '&' not in data:
        return token
    new_tokens = []
    for part in html5lib_shim.next_possible_entity(data):
        if not part:
            continue
        if part.startswith('&'):
            entity = html5lib_shim.match_entity(part)
            if entity is not None:
                if entity == 'amp':
                    new_tokens.append({'type': 'Characters', 'data': '&'})
                else:
                    new_tokens.append({'type': 'Entity', 'name': entity})
                remainder = part[len(entity) + 2:]
                if remainder:
                    new_tokens.append({'type': 'Characters', 'data': remainder})
                continue
        new_tokens.append({'type': 'Characters', 'data': part})
    return new_tokens