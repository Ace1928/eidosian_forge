def edns_from_text(text):
    """Convert a space-separated list of EDNS flag text values into a EDNS
    flags value.

    Returns an ``int``
    """
    return _from_text(text, _edns_by_text)