from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def all_variants(include_blunt_angles=True):
    """For testing and examples; yield all variants of all lattices."""
    a, b, c = (3.0, 4.0, 5.0)
    alpha = 55.0
    yield CUB(a)
    yield FCC(a)
    yield BCC(a)
    yield TET(a, c)
    bct1 = BCT(2 * a, c)
    bct2 = BCT(a, c)
    assert bct1.variant == 'BCT1'
    assert bct2.variant == 'BCT2'
    yield bct1
    yield bct2
    yield ORC(a, b, c)
    a0 = np.sqrt(1.0 / (1 / b ** 2 + 1 / c ** 2))
    orcf1 = ORCF(0.5 * a0, b, c)
    orcf2 = ORCF(1.2 * a0, b, c)
    orcf3 = ORCF(a0, b, c)
    assert orcf1.variant == 'ORCF1'
    assert orcf2.variant == 'ORCF2'
    assert orcf3.variant == 'ORCF3'
    yield orcf1
    yield orcf2
    yield orcf3
    yield ORCI(a, b, c)
    yield ORCC(a, b, c)
    yield HEX(a, c)
    rhl1 = RHL(a, alpha=55.0)
    assert rhl1.variant == 'RHL1'
    yield rhl1
    rhl2 = RHL(a, alpha=105.0)
    assert rhl2.variant == 'RHL2'
    yield rhl2
    yield MCL(a, b, c, alpha=70.0)
    mclc1 = MCLC(a, b, c, 80)
    assert mclc1.variant == 'MCLC1'
    yield mclc1
    mclc3 = MCLC(1.8 * a, b, c * 2, 80)
    assert mclc3.variant == 'MCLC3'
    yield mclc3
    mclc5 = MCLC(b, b, 1.1 * b, 70)
    assert mclc5.variant == 'MCLC5'
    yield mclc5

    def get_tri(kcellpar):
        icell = Cell.fromcellpar(kcellpar)
        cellpar = Cell(4 * icell.reciprocal()).cellpar()
        return TRI(*cellpar)
    tri1a = get_tri([1.0, 1.2, 1.4, 120.0, 110.0, 100.0])
    assert tri1a.variant == 'TRI1a'
    yield tri1a
    tri1b = get_tri([1.0, 1.2, 1.4, 50.0, 60.0, 70.0])
    assert tri1b.variant == 'TRI1b'
    yield tri1b
    tri2a = get_tri([1.0, 1.2, 1.4, 120.0, 110.0, 90.0])
    assert tri2a.variant == 'TRI2a'
    yield tri2a
    tri2b = get_tri([1.0, 1.2, 1.4, 50.0, 60.0, 90.0])
    assert tri2b.variant == 'TRI2b'
    yield tri2b
    yield OBL(a, b, alpha=alpha)
    yield RECT(a, b)
    yield CRECT(a, alpha=alpha)
    yield HEX2D(a)
    yield SQR(a)
    yield LINE(a)
    if include_blunt_angles:
        beta = 110
        yield OBL(a, b, alpha=beta)
        yield CRECT(a, alpha=beta)