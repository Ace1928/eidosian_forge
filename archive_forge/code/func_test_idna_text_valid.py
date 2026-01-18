@given(idna_text())
def test_idna_text_valid(self, text):
    """
            idna_text() generates IDNA-encodable text.
            """
    try:
        idna_encode(text)
    except IDNAError:
        raise AssertionError('Invalid IDNA text: {!r}'.format(text))