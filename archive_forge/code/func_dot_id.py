import collections
import hashlib
from cliff.formatters import base
@staticmethod
def dot_id(url, prefix=None):
    """Build an id with a prefix and a truncated hash of the URL"""
    if not url:
        return None
    if not prefix:
        prefix = 'r'
    hash_object = hashlib.sha256(url.encode('utf-8'))
    return '%s_%s' % (prefix, hash_object.hexdigest()[:20])