import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def calculate_digest(ip, timestamp, secret, userid, tokens, user_data, digest_algo):
    secret = maybe_encode(secret)
    userid = maybe_encode(userid)
    tokens = maybe_encode(tokens)
    user_data = maybe_encode(user_data)
    digest0 = maybe_encode(digest_algo(encode_ip_timestamp(ip, timestamp) + secret + userid + b'\x00' + tokens + b'\x00' + user_data).hexdigest())
    digest = digest_algo(digest0 + secret).hexdigest()
    return maybe_encode(digest)