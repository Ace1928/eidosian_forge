import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterable, Iterator
import ase.io
from ase.db import connect
from ase.db.core import convert_str_to_int_float_or_str
from ase.db.row import row2dct
from ase.db.table import Table, all_columns
from ase.utils import plural
def row2str(row) -> str:
    t = row2dct(row)
    S = [t['formula'] + ':', 'Unit cell in Ang:', 'axis|periodic|          x|          y|          z|' + '    length|     angle']
    c = 1
    fmt = '   {0}|     {1}|{2[0]:>11}|{2[1]:>11}|{2[2]:>11}|' + '{3:>10}|{4:>10}'
    for p, axis, L, A in zip(row.pbc, t['cell'], t['lengths'], t['angles']):
        S.append(fmt.format(c, [' no', 'yes'][p], axis, L, A))
        c += 1
    S.append('')
    if 'stress' in t:
        S += ['Stress tensor (xx, yy, zz, zy, zx, yx) in eV/Ang^3:', '   {}\n'.format(t['stress'])]
    if 'dipole' in t:
        S.append('Dipole moment in e*Ang: ({})\n'.format(t['dipole']))
    if 'constraints' in t:
        S.append('Constraints: {}\n'.format(t['constraints']))
    if 'data' in t:
        S.append('Data: {}\n'.format(t['data']))
    width0 = max(max((len(row[0]) for row in t['table'])), 3)
    width1 = max(max((len(row[1]) for row in t['table'])), 11)
    S.append('{:{}} | {:{}} | Value'.format('Key', width0, 'Description', width1))
    for key, desc, value in t['table']:
        S.append('{:{}} | {:{}} | {}'.format(key, width0, desc, width1, value))
    return '\n'.join(S)