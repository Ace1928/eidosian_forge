@given(hostname_labels())
def test_hostname_labels_valid_idn(self, label):
    """
            hostname_labels() generates IDN host name labels.
            """
    try:
        check_label(label)
        idna_encode(label)
    except UnicodeError:
        raise AssertionError('Invalid IDN label: {!r}'.format(label))