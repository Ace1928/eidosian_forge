import os
import re
import logging
import collections
import pyzor.account
def load_local_whitelist(filepath):
    """Load the local digest skip file."""
    if not os.path.exists(filepath):
        return set()
    whitelist = set()
    with open(filepath) as serverf:
        for line in serverf:
            line = _COMMENT_P.sub('', line).strip()
            if line:
                whitelist.add(line)
    return whitelist