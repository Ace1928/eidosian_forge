import re
import string
import warnings
from bleach._vendor.html5lib import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib import (
from bleach._vendor.html5lib.constants import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib.constants import (
from bleach._vendor.html5lib.filters.base import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib._inputstream import (
from bleach._vendor.html5lib.serializer import (
from bleach._vendor.html5lib._tokenizer import (
from bleach._vendor.html5lib._trie import (
def escape_base_amp(self, stoken):
    """Escapes just bare & in HTML attribute values"""
    stoken = stoken.replace('&amp;', '&')
    for part in next_possible_entity(stoken):
        if not part:
            continue
        if part.startswith('&'):
            entity = match_entity(part)
            if entity is not None and convert_entity(entity) is not None:
                yield f'&{entity};'
                part = part[len(entity) + 2:]
                if part:
                    yield part
                continue
        yield part.replace('&', '&amp;')