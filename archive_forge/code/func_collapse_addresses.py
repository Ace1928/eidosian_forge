import functools
def collapse_addresses(addresses):
    """Collapse a list of IP objects.

    Example:
        collapse_addresses([IPv4Network('192.0.2.0/25'),
                            IPv4Network('192.0.2.128/25')]) ->
                           [IPv4Network('192.0.2.0/24')]

    Args:
        addresses: An iterator of IPv4Network or IPv6Network objects.

    Returns:
        An iterator of the collapsed IPv(4|6)Network objects.

    Raises:
        TypeError: If passed a list of mixed version objects.

    """
    addrs = []
    ips = []
    nets = []
    for ip in addresses:
        if isinstance(ip, _BaseAddress):
            if ips and ips[-1]._version != ip._version:
                raise TypeError('%s and %s are not of the same version' % (ip, ips[-1]))
            ips.append(ip)
        elif ip._prefixlen == ip._max_prefixlen:
            if ips and ips[-1]._version != ip._version:
                raise TypeError('%s and %s are not of the same version' % (ip, ips[-1]))
            try:
                ips.append(ip.ip)
            except AttributeError:
                ips.append(ip.network_address)
        else:
            if nets and nets[-1]._version != ip._version:
                raise TypeError('%s and %s are not of the same version' % (ip, nets[-1]))
            nets.append(ip)
    ips = sorted(set(ips))
    if ips:
        for first, last in _find_address_range(ips):
            addrs.extend(summarize_address_range(first, last))
    return _collapse_addresses_internal(addrs + nets)