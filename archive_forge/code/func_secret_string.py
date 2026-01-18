import base64
import os
import random
def secret_string(length=25):
    """
    Returns a random string of the given length.  The string
    is a base64-encoded version of a set of random bytes, truncated
    to the given length (and without any newlines).
    """
    s = random_bytes(length)
    s = base64.b64encode(s)
    s = s.decode('ascii')
    for badchar in '\n\r=':
        s = s.replace(badchar, '')
    return s[:length]