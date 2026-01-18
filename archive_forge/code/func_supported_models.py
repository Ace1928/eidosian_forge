from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
@classmethod
def supported_models(cls):
    return [getattr(cls, item) for item in dir(cls) if item.startswith('FTD_')]