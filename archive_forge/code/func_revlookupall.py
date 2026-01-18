from . import Base
from . Base import ServerError
def revlookupall(name, timeout=30):
    """convenience routine for doing a reverse lookup of an address"""
    a = name.split('.')
    a.reverse()
    b = '.'.join(a) + '.in-addr.arpa'
    qtype = 'ptr'
    names = dnslookup(b, qtype, timeout)
    names.sort(key=str.__len__)
    return names