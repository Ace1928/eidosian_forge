from __future__ import absolute_import
@composite
def idna_text(draw, min_size=1, max_size=None):
    """
        A strategy which generates IDNA-encodable text.

        @param min_size: The minimum number of characters in the text.
            C{None} is treated as C{0}.

        @param max_size: The maximum number of characters in the text.
            Use C{None} for an unbounded size.
        """
    alphabet = idna_characters()
    assert min_size >= 1
    if max_size is not None:
        assert max_size >= 1
    result = cast(Text, draw(text(min_size=min_size, max_size=max_size, alphabet=alphabet)))
    try:
        idna_encode(result)
    except IDNAError:
        assume(False)
    return result