from __future__ import annotations
import re
from copy import deepcopy
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.core.composition import Composition, reduce_formula
from pymatgen.util.string import Stringify, charge_string, formula_double_format
@classmethod
def from_formula(cls, formula: str) -> Self:
    """Creates Ion from formula. The net charge can either be represented as
        Mn++, Mn+2, Mn[2+], Mn[++], or Mn[+2]. Note the order of the sign and
        magnitude in each representation.

        Also note that (aq) can be included in the formula, e.g. "NaOH (aq)".

        Args:
            formula (str): The formula to create ion from.

        Returns:
            Ion
        """
    charge = 0.0
    f = formula
    m = re.search('\\(aq\\)', f)
    if m:
        f = f.replace(m.group(), '', 1)
    m = re.search('\\[([^\\[\\]]+)\\]', f)
    if m:
        m_chg = re.search('([\\.\\d]*)([+-]*)([\\.\\d]*)', m.group(1))
        if m_chg:
            if m_chg.group(1) != '':
                if m_chg.group(3) != '':
                    raise ValueError('Invalid formula')
                charge += float(m_chg.group(1)) * float(m_chg.group(2) + '1')
            elif m_chg.group(3) != '':
                charge += float(m_chg.group(3)) * float(m_chg.group(2) + '1')
            else:
                for i in re.findall('[+-]', m_chg.group(2)):
                    charge += float(i + '1')
        f = f.replace(m.group(), '', 1)
    for m_chg in re.finditer('([+-])([\\.\\d]*)', f):
        sign = m_chg.group(1)
        sgn = float(str(sign + '1'))
        if m_chg.group(2).strip() != '':
            charge += float(m_chg.group(2)) * sgn
        else:
            charge += sgn
        f = f.replace(m_chg.group(), '', 1)
    composition = Composition(f)
    return cls(composition, charge)