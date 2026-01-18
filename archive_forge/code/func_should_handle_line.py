from __future__ import print_function
import re
import hashlib
@classmethod
def should_handle_line(cls, s):
    return len(s) and cls.min_line_length <= len(s)