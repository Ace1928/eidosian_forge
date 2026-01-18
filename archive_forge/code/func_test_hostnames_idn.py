@given(hostnames())
def test_hostnames_idn(self, hostname):
    """
            hostnames() generates a IDN host names.
            """
    try:
        for label in hostname.split(u'.'):
            check_label(label)
        idna_encode(hostname)
    except UnicodeError:
        raise AssertionError('Invalid IDN host name: {!r}'.format(hostname))