@given(hostname_labels(allow_idn=False))
def test_hostname_labels_valid_ascii(self, label):
    """
            hostname_labels() generates a ASCII host name labels.
            """
    try:
        check_label(label)
        label.encode('ascii')
    except UnicodeError:
        raise AssertionError('Invalid ASCII label: {!r}'.format(label))