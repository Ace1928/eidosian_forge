def decode_base64_like_DT_code(chars):
    """
    Given a base64-like encoding, return the DT code and, if present in the
    encoding as well, return the flips, otherwise None.
    """
    code, pos = _decode_DT_code(chars)
    num_crossings = sum((len(component) for component in code))
    flips = _decode_flips(chars[pos:])[:num_crossings]
    return (code, _empty_to_none(flips))