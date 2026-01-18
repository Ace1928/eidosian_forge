from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
@property
def duplicate_params(self):
    seen_keys = collections.defaultdict(int)
    all_keys = (p[0] for p in (self.decoded_body or []) + self.uri_query_params)
    for k in all_keys:
        seen_keys[k] += 1
    return [k for k, c in seen_keys.items() if c > 1]