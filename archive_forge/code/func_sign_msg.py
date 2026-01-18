import time
import hashlib
import pyzor
def sign_msg(hashed_key, timestamp, msg, hash_=hashlib.sha1):
    """Converts the key, timestamp (epoch seconds), and msg into a digest.

    lower(H(H(M) + ':' T + ':' + K))
    M is message
    T is integer epoch timestamp
    K is hashed_key
    H is the hash function (currently SHA1)
    """
    msg = msg.as_string().strip().encode('utf8')
    digest = hash_()
    digest.update(hash_(msg).digest())
    digest.update((':%d:%s' % (timestamp, hashed_key)).encode('utf8'))
    return digest.hexdigest().lower()