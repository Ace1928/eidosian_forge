from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
def getSupportedKeyExchanges():
    """
    Get a list of supported key exchange algorithm names in order of
    preference.

    @return: A C{list} of supported key exchange algorithm names.
    @rtype: C{list} of L{bytes}
    """
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import ec
    from twisted.conch.ssh.keys import _curveTable
    backend = default_backend()
    kexAlgorithms = _kexAlgorithms.copy()
    for keyAlgorithm in list(kexAlgorithms):
        if keyAlgorithm.startswith(b'ecdh'):
            keyAlgorithmDsa = keyAlgorithm.replace(b'ecdh', b'ecdsa')
            supported = backend.elliptic_curve_exchange_algorithm_supported(ec.ECDH(), _curveTable[keyAlgorithmDsa])
        elif keyAlgorithm.startswith(b'curve25519-sha256'):
            supported = backend.x25519_supported()
        else:
            supported = True
        if not supported:
            kexAlgorithms.pop(keyAlgorithm)
    return sorted(kexAlgorithms, key=lambda kexAlgorithm: kexAlgorithms[kexAlgorithm].preference)