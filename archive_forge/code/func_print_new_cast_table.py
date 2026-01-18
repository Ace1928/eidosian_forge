import numpy as np
from collections import namedtuple
def print_new_cast_table(*, can_cast=True, legacy=False, flags=False):
    """Prints new casts, the values given are default "can-cast" values, not
    actual ones.
    """
    from numpy.core._multiarray_tests import get_all_cast_information
    cast_table = {-1: ' ', 0: '#', 1: '#', 2: '=', 3: '~', 4: '.'}
    flags_table = {0: '▗', 7: '█', 1: '▚', 2: '▐', 4: '▄', 3: '▜', 5: '▙', 6: '▟'}
    cast_info = namedtuple('cast_info', ['can_cast', 'legacy', 'flags'])
    no_cast_info = cast_info(' ', ' ', ' ')
    casts = get_all_cast_information()
    table = {}
    dtypes = set()
    for cast in casts:
        dtypes.add(cast['from'])
        dtypes.add(cast['to'])
        if cast['from'] not in table:
            table[cast['from']] = {}
        to_dict = table[cast['from']]
        can_cast = cast_table[cast['casting']]
        legacy = 'L' if cast['legacy'] else '.'
        flags = 0
        if cast['requires_pyapi']:
            flags |= 1
        if cast['supports_unaligned']:
            flags |= 2
        if cast['no_floatingpoint_errors']:
            flags |= 4
        flags = flags_table[flags]
        to_dict[cast['to']] = cast_info(can_cast=can_cast, legacy=legacy, flags=flags)
    types = np.typecodes['All']

    def sorter(x):
        dtype = np.dtype(x.type)
        try:
            indx = types.index(dtype.char)
        except ValueError:
            indx = np.inf
        return (indx, dtype.char)
    dtypes = sorted(dtypes, key=sorter)

    def print_table(field='can_cast'):
        print('X', end=' ')
        for dt in dtypes:
            print(np.dtype(dt.type).char, end=' ')
        print()
        for from_dt in dtypes:
            print(np.dtype(from_dt.type).char, end=' ')
            row = table.get(from_dt, {})
            for to_dt in dtypes:
                print(getattr(row.get(to_dt, no_cast_info), field), end=' ')
            print()
    if can_cast:
        print()
        print('Casting: # is equivalent, = is safe, ~ is same-kind, and . is unsafe')
        print()
        print_table('can_cast')
    if legacy:
        print()
        print('L denotes a legacy cast . a non-legacy one.')
        print()
        print_table('legacy')
    if flags:
        print()
        print(f'{flags_table[0]}: no flags, {flags_table[1]}: PyAPI, {flags_table[2]}: supports unaligned, {flags_table[4]}: no-float-errors')
        print()
        print_table('flags')