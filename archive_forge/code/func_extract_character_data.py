from __future__ import unicode_literals
import re
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def extract_character_data(self, token_list):
    """Extracts and squashes character sequences in a token stream"""
    out = []
    for token in token_list:
        token_type = token['type']
        if token_type in ['Characters', 'SpaceCharacters']:
            out.append(token['data'])
    return u''.join(out)