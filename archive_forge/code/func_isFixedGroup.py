from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
def isFixedGroup(kexAlgorithm):
    """
    Returns C{True} if C{kexAlgorithm} has a fixed prime / generator group.

    @param kexAlgorithm: The key exchange algorithm name.
    @type kexAlgorithm: L{bytes}

    @return: C{True} if C{kexAlgorithm} has a fixed prime / generator group,
        otherwise C{False}.
    @rtype: L{bool}
    """
    return _IFixedGroupKexAlgorithm.providedBy(getKex(kexAlgorithm))