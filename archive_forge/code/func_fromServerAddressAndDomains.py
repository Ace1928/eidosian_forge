from twisted.application import service
from twisted.internet import defer, task
from twisted.names import client, common, dns, resolve
from twisted.names.authority import FileAuthority
from twisted.python import failure, log
from twisted.python.compat import nativeString
@classmethod
def fromServerAddressAndDomains(cls, serverAddress, domains):
    """
        Construct a new L{SecondaryAuthorityService} from a tuple giving a
        server address and a C{str} giving the name of a domain for which this
        is an authority.

        @param serverAddress: A two-tuple, the first element of which is a
            C{str} giving an IP address and the second element of which is a
            C{int} giving a port number.  Together, these define where zone
            transfers will be attempted from.

        @param domains: Domain names for which to perform zone transfers.
        @type domains: sequence of L{bytes}

        @return: A new instance of L{SecondaryAuthorityService}.
        """
    primary, port = serverAddress
    service = cls(primary, [])
    service._port = port
    service.domains = [SecondaryAuthority.fromServerAddressAndDomain(serverAddress, d) for d in domains]
    return service