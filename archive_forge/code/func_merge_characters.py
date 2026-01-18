from itertools import chain
import re
import warnings
from xml.sax.saxutils import unescape
from bleach import html5lib_shim
from bleach import parse_shim
def merge_characters(self, token_iterator):
    """Merge consecutive Characters tokens in a stream"""
    characters_buffer = []
    for token in token_iterator:
        if characters_buffer:
            if token['type'] == 'Characters':
                characters_buffer.append(token)
                continue
            else:
                new_token = {'data': ''.join([char_token['data'] for char_token in characters_buffer]), 'type': 'Characters'}
                characters_buffer = []
                yield new_token
        elif token['type'] == 'Characters':
            characters_buffer.append(token)
            continue
        yield token
    new_token = {'data': ''.join([char_token['data'] for char_token in characters_buffer]), 'type': 'Characters'}
    yield new_token