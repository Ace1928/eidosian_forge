from twisted.internet import defer
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.names import common, dns
from twisted.python import failure
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
def searchFileForAll(hostsFile, name):
    """
    Search the given file, which is in hosts(5) standard format, for addresses
    associated with a given name.

    @param hostsFile: The name of the hosts(5)-format file to search.
    @type hostsFile: L{FilePath}

    @param name: The name to search for.
    @type name: C{bytes}

    @return: L{None} if the name is not found in the file, otherwise a
        C{str} giving the address in the file associated with the name.
    """
    results = []
    try:
        lines = hostsFile.getContent().splitlines()
    except BaseException:
        return results
    name = name.lower()
    for line in lines:
        idx = line.find(b'#')
        if idx != -1:
            line = line[:idx]
        if not line:
            continue
        parts = line.split()
        if name.lower() in [s.lower() for s in parts[1:]]:
            try:
                maybeIP = nativeString(parts[0])
            except ValueError:
                continue
            if isIPAddress(maybeIP) or isIPv6Address(maybeIP):
                results.append(maybeIP)
    return results