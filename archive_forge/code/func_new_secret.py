import hmac, base64, random, time, warnings
from functools import reduce
from paste.request import get_cookies
def new_secret():
    """ returns a 64 byte secret """
    secret = ''.join(random.sample(_all_chars, 64))
    secret = secret.encode('utf8')
    return secret