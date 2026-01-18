from base64 import b64encode
from binascii import b2a_base64, a2b_base64
def body_encode(s, maxlinelen=76, eol=NL):
    """Encode a string with base64.

    Each line will be wrapped at, at most, maxlinelen characters (defaults to
    76 characters).

    Each line of encoded text will end with eol, which defaults to "\\n".  Set
    this to "\\r\\n" if you will be using the result of this function directly
    in an email.
    """
    if not s:
        return ''
    encvec = []
    max_unencoded = maxlinelen * 3 // 4
    for i in range(0, len(s), max_unencoded):
        enc = b2a_base64(s[i:i + max_unencoded]).decode('ascii')
        if enc.endswith(NL) and eol != NL:
            enc = enc[:-1] + eol
        encvec.append(enc)
    return EMPTYSTRING.join(encvec)