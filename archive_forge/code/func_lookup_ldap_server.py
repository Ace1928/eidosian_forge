import typing as t
def lookup_ldap_server() -> t.Tuple[str, int]:
    """Attempts to lookup LDAP server.

    Attempts to lookup LDAP server based on the current Kerberos host
    configuration. Will them perform an SRV lookup for
    '_ldap._tcp.dc._msdcs.{realm}' to get the LDAP server hostname nad port.

    Returns:
        Tuple[str, int]: The LDAP hostname and port.

    Raises:
        ImportError: Missing krb5 or dnspython.
        krb5.Krb5Error: Kerberos configuration problem
        dns.exception.DNSException: DNS lookup error.
    """
    required_libs = [(HAS_KRB5, 'krb5'), (HAS_DNSPYTHON, 'dnspython')]
    missing_libs = [lib for present, lib in required_libs if not present]
    if missing_libs:
        raise ImportError(f'Cannot lookup server without the python libraries {', '.join(missing_libs)}')
    ctx = krb5.init_context()
    default_realm = krb5.get_default_realm(ctx).decode('utf-8')
    answer = SrvRecord.lookup('ldap', 'tcp', f'dc._msdcs.{default_realm}')[0]
    return (answer.target, answer.port)