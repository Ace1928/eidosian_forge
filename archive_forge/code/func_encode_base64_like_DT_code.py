def encode_base64_like_DT_code(code, flips=None):
    """
    Given a DT code and optionally flips, convert to base64-like encoding.
    """
    if flips:
        return _encode_DT_code(code) + _encode_flips(flips)
    else:
        return _encode_DT_code(code)