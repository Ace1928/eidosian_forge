from collections import OrderedDict
from itertools import chain
def all_original_names(group=None, not_group=None, only_aliased=False, only_CET=False):
    """
    Returns a list (optionally filtered) of the names of the available colormaps
    Filters available:
    - group: only include maps whose name include the given string(s)
      (e.g. "'linear'" or "['linear','diverging']").
    - not_group: filter out any maps whose names include the given string(s)
    - only_aliased: only include maps with shorter/simpler aliases
    - only_CET: only include maps from CET
    """
    names = palette.keys()
    if group:
        groups = group if isinstance(group, list) else [group]
        names = [n for ns in [list(filter(lambda x: g in x, names)) for g in groups] for n in ns]
    if not_group:
        not_groups = not_group if isinstance(not_group, list) else [not_group]
        for g in not_groups:
            names = list(filter(lambda x: g not in x, names))
    if only_aliased:
        names = filter(lambda x: x in aliases.keys(), names)
    else:
        names = filter(lambda x: x not in chain.from_iterable(aliases.values()), names)
    if only_CET:
        names = filter(lambda x: x in cetnames_flipped.values(), names)
    else:
        names = filter(lambda x: x not in cetnames_flipped.values(), names)
    return sorted(list(names))