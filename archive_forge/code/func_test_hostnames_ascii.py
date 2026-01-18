@given(hostnames(allow_idn=False))
def test_hostnames_ascii(self, hostname):
    """
            hostnames() generates a ASCII host names.
            """
    try:
        for label in hostname.split(u'.'):
            check_label(label)
        hostname.encode('ascii')
    except UnicodeError:
        raise AssertionError('Invalid ASCII host name: {!r}'.format(hostname))