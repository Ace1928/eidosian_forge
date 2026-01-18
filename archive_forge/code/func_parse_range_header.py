def parse_range_header(header_value, valid_units=('bytes', 'none')):
    """Parses the value of an HTTP Range: header.

    The value of the header as a string should be passed in; without
    the header name itself.

    Returns a range_set object.
    """
    ranges, k = parse_range_set(header_value, valid_units=valid_units)
    if k < len(header_value):
        raise ParseError('Range header has unexpected or unparsable characters', header_value, k)
    return ranges