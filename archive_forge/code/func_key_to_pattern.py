from __future__ import annotations
import re
from kombu.utils.text import escape_regex
def key_to_pattern(self, rkey):
    """Get the corresponding regex for any routing key."""
    return '^%s$' % '\\.'.join((self.wildcards.get(word, word) for word in escape_regex(rkey, '.#*').split('.')))