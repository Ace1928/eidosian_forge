def parse_accept_language_header(header_value):
    """Parses the Accept-Language header.

    Returns a list of tuples, each like:

        (language_tag, qvalue, accept_parameters)

    """
    alist, k = parse_qvalue_accept_list(header_value)
    if k < len(header_value):
        raise ParseError('Accept-Language header is invalid', header_value, k)
    langlist = []
    for token, langparms, q, acptparms in alist:
        if langparms:
            raise ParseError('Language tag may not have any parameters', header_value, 0)
        lang = language_tag(token)
        langlist.append((lang, q, acptparms))
    return langlist