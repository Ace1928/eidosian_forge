import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
def canonical_query_str(verb, params):
    canonical_qs = ''
    if verb.upper() != 'POST':
        canonical_qs = self._canonical_qs(params)
    return canonical_qs