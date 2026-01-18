from netaddr.core import AddrFormatError, AddrConversionError
from netaddr.ip import IPRange, IPAddress, IPNetwork, iprange_to_cidrs
def iprange_to_globs(start, end):
    """
    A function that accepts an arbitrary start and end IP address or subnet
    and returns one or more glob-style IP ranges.

    :param start: the start IP address or subnet.

    :param end: the end IP address or subnet.

    :return: a list containing one or more IP globs.
    """
    start = IPAddress(start)
    end = IPAddress(end)
    if start.version != 4 and end.version != 4:
        raise AddrConversionError('IP glob ranges only support IPv4!')

    def _iprange_to_glob(lb, ub):
        t1 = [int(_) for _ in str(lb).split('.')]
        t2 = [int(_) for _ in str(ub).split('.')]
        tokens = []
        seen_hyphen = False
        seen_asterisk = False
        for i in range(4):
            if t1[i] == t2[i]:
                tokens.append(str(t1[i]))
            elif t1[i] == 0 and t2[i] == 255:
                tokens.append('*')
                seen_asterisk = True
            elif not seen_asterisk:
                if not seen_hyphen:
                    tokens.append('%s-%s' % (t1[i], t2[i]))
                    seen_hyphen = True
                else:
                    raise AddrConversionError('only 1 hyphenated octet per IP glob allowed!')
            else:
                raise AddrConversionError('asterisks are not allowed before hyphenated octets!')
        return '.'.join(tokens)
    globs = []
    try:
        ipglob = _iprange_to_glob(start, end)
        if not valid_glob(ipglob):
            raise AddrConversionError('invalid ip glob created')
        globs.append(ipglob)
    except AddrConversionError:
        for cidr in iprange_to_cidrs(start, end):
            ipglob = _iprange_to_glob(cidr[0], cidr[-1])
            globs.append(ipglob)
    return globs