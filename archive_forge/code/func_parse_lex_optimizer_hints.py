def parse_lex_optimizer_hints(f):
    """Parse optimizer hints in lex.h."""
    results = set()
    for m in re.finditer('{SYM_H\\("(?P<keyword>[a-z0-9_]+)",', f, flags=re.I):
        results.add(m.group('keyword').lower())
    if not results:
        raise ValueError('No optimizer hints found')
    return results