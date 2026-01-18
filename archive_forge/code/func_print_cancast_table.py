import numpy as np
from collections import namedtuple
def print_cancast_table(ntypes):
    print('X', end=' ')
    for char in ntypes:
        print(char, end=' ')
    print()
    for row in ntypes:
        print(row, end=' ')
        for col in ntypes:
            if np.can_cast(row, col, 'equiv'):
                cast = '#'
            elif np.can_cast(row, col, 'safe'):
                cast = '='
            elif np.can_cast(row, col, 'same_kind'):
                cast = '~'
            elif np.can_cast(row, col, 'unsafe'):
                cast = '.'
            else:
                cast = ' '
            print(cast, end=' ')
        print()