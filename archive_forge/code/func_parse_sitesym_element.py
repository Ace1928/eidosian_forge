import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def parse_sitesym_element(element):
    """Parses one element from a single site symmetry in the form used
    by the International Tables.
    
    Examples:
    
    >>> parse_sitesym_element("x")
    ([(0, 1)], 0.0)
    >>> parse_sitesym_element("-1/2-y")
    ([(1, -1)], -0.5)
    >>> parse_sitesym_element("z+0.25")
    ([(2, 1)], 0.25)
    >>> parse_sitesym_element("x-z+0.5")
    ([(0, 1), (2, -1)], 0.5)
    
    
    
    Parameters
    ----------
    
    element: str
      Site symmetry like "x" or "-y+1/4" or "0.5+z".
      
      
    Returns
    -------
    
    list[tuple[int, int]]
      Rotation information in the form '(index, sign)' where index is
      0 for "x", 1 for "y" and 2 for "z" and sign is '1' for a positive
      entry and '-1' for a negative entry. E.g. "x" is '(0, 1)' and
      "-z" is (2, -1).
      
    float
      Translation information in fractional space. E.g. "-1/4" is
      '-0.25' and "1/2" is '0.5' and "0.75" is '0.75'.
    
    
    """
    element = element.lower()
    is_positive = True
    is_frac = False
    sng_trans = None
    fst_trans = []
    snd_trans = []
    rot = []
    for char in element:
        if char == '+':
            is_positive = True
        elif char == '-':
            is_positive = False
        elif char == '/':
            is_frac = True
        elif char in 'xyz':
            rot.append((ord(char) - ord('x'), 1 if is_positive else -1))
        elif char.isdigit() or char == '.':
            if sng_trans is None:
                sng_trans = 1.0 if is_positive else -1.0
            if is_frac:
                snd_trans.append(char)
            else:
                fst_trans.append(char)
    trans = 0.0 if not fst_trans else sng_trans * float(''.join(fst_trans))
    if is_frac:
        trans /= float(''.join(snd_trans))
    return (rot, trans)