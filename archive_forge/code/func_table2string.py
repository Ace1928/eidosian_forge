import re
from io import StringIO
from math import ceil
import numpy as np
def table2string(table, out=None):
    """Given list of lists figure out their common widths and print to out

    Parameters
    ----------
    table : list of lists of strings
      What is aimed to be printed
    out : None or stream
      Where to print. If None -- will print and return string

    Returns
    -------
    string if out was None
    """
    print2string = out is None
    if print2string:
        out = StringIO()
    nelements_max = len(table) and max((len(x) for x in table))
    for i, table_ in enumerate(table):
        table[i] += [''] * (nelements_max - len(table_))
    atable = np.asarray(table)
    markup_strip = re.compile('^@([lrc]|w.*)')
    col_width = [max((len(markup_strip.sub('', x)) for x in column)) for column in atable.T]
    string = ''
    for i, table_ in enumerate(table):
        string_ = ''
        for j, item in enumerate(table_):
            item = str(item)
            if item.startswith('@'):
                align = item[1]
                item = item[2:]
                if align not in ('l', 'r', 'c', 'w'):
                    raise ValueError(f'Unknown alignment {align}. Known are l,r,c')
            else:
                align = 'c'
            nspacesl = max(ceil((col_width[j] - len(item)) / 2.0), 0)
            nspacesr = max(col_width[j] - nspacesl - len(item), 0)
            if align in ('w', 'c'):
                pass
            elif align == 'l':
                nspacesl, nspacesr = (0, nspacesl + nspacesr)
            elif align == 'r':
                nspacesl, nspacesr = (nspacesl + nspacesr, 0)
            else:
                raise RuntimeError(f'Should not get here with align={align}')
            string_ += '%%%ds%%s%%%ds ' % (nspacesl, nspacesr) % ('', item, '')
        string += string_.rstrip() + '\n'
    out.write(string)
    if print2string:
        value = out.getvalue()
        out.close()
        return value