from binascii import hexlify
from hashlib import md5, sha1
def calcResponse(HA1, HA2, algo, pszNonce, pszNonceCount, pszCNonce, pszQop):
    """
    Compute the digest for the given parameters.

    @param HA1: The H(A1) value, as computed by L{calcHA1}.
    @param HA2: The H(A2) value, as computed by L{calcHA2}.
    @param pszNonce: The challenge nonce.
    @param pszNonceCount: The (client) nonce count value for this response.
    @param pszCNonce: The client nonce.
    @param pszQop: The Quality-of-Protection value.
    """
    m = algorithms[algo]()
    m.update(HA1)
    m.update(b':')
    m.update(pszNonce)
    m.update(b':')
    if pszNonceCount and pszCNonce:
        m.update(pszNonceCount)
        m.update(b':')
        m.update(pszCNonce)
        m.update(b':')
        m.update(pszQop)
        m.update(b':')
    m.update(HA2)
    respHash = hexlify(m.digest())
    return respHash