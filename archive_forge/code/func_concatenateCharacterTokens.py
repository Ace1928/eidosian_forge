from __future__ import absolute_import, division, unicode_literals
from .. import constants
from .._utils import default_etree
def concatenateCharacterTokens(tokens):
    pendingCharacters = []
    for token in tokens:
        type = token['type']
        if type in ('Characters', 'SpaceCharacters'):
            pendingCharacters.append(token['data'])
        else:
            if pendingCharacters:
                yield {'type': 'Characters', 'data': ''.join(pendingCharacters)}
                pendingCharacters = []
            yield token
    if pendingCharacters:
        yield {'type': 'Characters', 'data': ''.join(pendingCharacters)}