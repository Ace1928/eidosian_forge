import re
def filter_classes(classes, regex=_mf2_classes_re):
    """detect classes that are valid names for mf2, sort in dictionary by prefix"""
    types = {x: set() for x in ('u', 'p', 'dt', 'e', 'h')}
    for c in classes:
        match = regex.match(c)
        if match:
            if c[0] == 'h':
                types['h'].add(c)
            else:
                types[match.group(1)].add(match.group(2))
    return types