import sys
def resolve_bases(bases):
    """Resolve MRO entries dynamically as specified by PEP 560."""
    new_bases = list(bases)
    updated = False
    shift = 0
    for i, base in enumerate(bases):
        if isinstance(base, type):
            continue
        if not hasattr(base, '__mro_entries__'):
            continue
        new_base = base.__mro_entries__(bases)
        updated = True
        if not isinstance(new_base, tuple):
            raise TypeError('__mro_entries__ must return a tuple')
        else:
            new_bases[i + shift:i + shift + 1] = new_base
            shift += len(new_base) - 1
    if not updated:
        return bases
    return tuple(new_bases)