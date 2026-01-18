from __future__ import unicode_literals
import re
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def handle_a_tag(self, token_buffer):
    """Handle the "a" tag

        This could adjust the link or drop it altogether depending on what the
        callbacks return.

        This yields the new set of tokens.

        """
    a_token = token_buffer[0]
    if a_token['data']:
        attrs = a_token['data']
    else:
        attrs = {}
    text = self.extract_character_data(token_buffer)
    attrs['_text'] = text
    attrs = self.apply_callbacks(attrs, False)
    if attrs is None:
        yield {'type': 'Characters', 'data': text}
    else:
        new_text = attrs.pop('_text', '')
        a_token['data'] = alphabetize_attributes(attrs)
        if text == new_text:
            yield a_token
            for mem in token_buffer[1:]:
                yield mem
        else:
            yield a_token
            yield {'type': 'Characters', 'data': force_unicode(new_text)}
            yield token_buffer[-1]