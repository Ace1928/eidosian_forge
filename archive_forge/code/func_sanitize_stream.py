from itertools import chain
import re
import warnings
from xml.sax.saxutils import unescape
from bleach import html5lib_shim
from bleach import parse_shim
def sanitize_stream(self, token_iterator):
    for token in token_iterator:
        ret = self.sanitize_token(token)
        if not ret:
            continue
        if isinstance(ret, list):
            yield from ret
        else:
            yield ret