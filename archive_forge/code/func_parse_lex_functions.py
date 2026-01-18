def parse_lex_functions(f):
    """Parse MySQL function names from lex.h."""
    results = set()
    for m in re.finditer('{SYM_FN?\\("(?P<function>[a-z0-9_]+)",', f, flags=re.I):
        results.add(m.group('function').lower())
    if not results:
        raise ValueError('No lex functions found')
    return results