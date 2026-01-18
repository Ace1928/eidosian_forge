import numpy as np
from ase.data import atomic_numbers
from ase.ga.offspring_creator import OffspringCreator
def mendeleiev_table():
    """
        Returns the mendeleiev table as a python list of lists.
        Each cell contains either None or a pair (symbol, atomic number),
        or a list of pairs for the cells \\* and \\**.
    """
    import re
    elems = 'HHeLiBeBCNOFNeNaMgAlSiPSClArKCaScTiVCrMnFeCoNiCuZnGaGeAsSeBrKrRb'
    elems += 'SrYZrNbMoTcRuRhPdAgCdInSnSbTeIXeCsBaLaCePrNdPmSmEuGdTbDyHoErTm'
    elems += 'YbLuHfTaWReOsIrPtAuHgTlPbBiPoAtRnFrRaAcThPaUNpPuAmCmBkCfEsFmMd'
    elems += 'NoLrRfDbSgBhHsMtDsRgUubUutUuqUupUuhUusUuo'
    L = [(e, i + 1) for i, e in enumerate(re.compile('[A-Z][a-z]*').findall(elems))]
    for i, j in ((88, 103), (56, 71)):
        L[i] = L[i:j]
        L[i + 1:] = L[j:]
    for i, j in ((12, 10), (4, 10), (1, 16)):
        L[i:i] = [None] * j
    return [L[18 * i:18 * (i + 1)] for i in range(7)]