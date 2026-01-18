def desired_matches(desired, header):
    """Takes a list of desired mime-types in the order the server prefers to
    send them regardless of the browsers preference.

    Browsers (such as Firefox) technically want XML over HTML depending on how
    one reads the specification. This function is provided for a server to
    declare a set of desired mime-types it supports, and returns a subset of
    the desired list in the same order should each one be Accepted by the
    browser.

    >>> desired_matches(['text/html', 'application/xml'],     ...     'text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png')
    ['text/html', 'application/xml']
    >>> desired_matches(['text/html', 'application/xml'], 'application/xml,application/json')
    ['application/xml']
    """
    parsed_ranges = list(map(parse_media_range, header.split(',')))
    return [mimetype for mimetype in desired if quality_parsed(mimetype, parsed_ranges)]