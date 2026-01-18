import re
def allMatches(source, regex):
    """Return a list of matches for regex in source
    """
    pos = 0
    end = len(source)
    rv = []
    match = regex.search(source, pos)
    while match:
        rv.append(match)
        match = regex.search(source, match.end())
    return rv