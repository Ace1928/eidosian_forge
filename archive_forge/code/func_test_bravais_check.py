import numpy as np
import pytest
from ase.cell import Cell
from ase.lattice import bravais_lattices, UnsupportedLattice
from ase.build import bulk, fcc111
def test_bravais_check():
    bravais = {}
    for name in bravais_lattices:
        bravais[name.lower()] = bravais_lattices[name]

    def check_single(name, cell, pbc=None):
        c = Cell(cell)
        try:
            print('TEST', c, pbc)
            if pbc[:2].all() or sum(pbc) == 1:
                lattice = c.get_bravais_lattice(pbc=pbc)
            else:
                with pytest.raises(UnsupportedLattice):
                    lattice = c.get_bravais_lattice(pbc=pbc)
                return
        except RuntimeError:
            print('error checking {}'.format(name))
            raise
        name1 = lattice.name.lower()
        latname = name.split('@')[0]
        ok = latname == name1
        print(name, '-->', name1, 'OK' if ok else 'ERR', c.cellpar())
        assert ok, 'Expected {} but found {}'.format(latname, name1)

    def check(name, cell, pbc=None):
        if pbc is None:
            pbc = cell.any(1)
        pbc = np.asarray(pbc)
        cell = Cell(cell)
        check_single(name + '@012', cell[[0, 1, 2]], pbc=pbc[[0, 1, 2]])
        if cell.rank == 3 and pbc.sum() != 1:
            check_single(name + '@201', cell[[2, 0, 1]], pbc=pbc[[2, 0, 1]])
            check_single(name + '@120', cell[[1, 2, 0]], pbc=pbc[[1, 2, 0]])
    check('cub', bravais['cub'](3.3).tocell())
    check('fcc', bravais['fcc'](3.4).tocell())
    check('fcc', bulk('Au').cell)
    check('bcc', bravais['bcc'](3.5).tocell())
    check('bcc', bulk('Fe').cell)
    check('tet', bravais['tet'](4.0, 5.0).tocell())
    check('tet', np.diag([4.0, 5.0, 5.0]))
    check('tet', np.diag([5.0, 4.0, 5.0]))
    check('tet', np.diag([5.0, 5.0, 4.0]))
    check('bct', bravais['bct'](3.0, 4.0).tocell())
    check('orc', bravais['orc'](3.0, 4.0, 5.0).tocell())
    check('orcf', bravais['orcf'](4.0, 5.0, 7.0).tocell())
    check('orci', bravais['orci'](2.0, 5.0, 6.0).tocell())
    check('orcc', bravais['orcc'](3.0, 4.0, 5.0).tocell())
    check('hex', fcc111('Au', size=(1, 1, 3), periodic=True).cell)
    check('hex', bravais['hex'](5.0, 6.0).tocell())
    check('rhl', bravais['rhl'](4.0, 54.0).tocell())
    check('mcl', bravais['mcl'](2.0, 3.0, 4.0, 62.0).tocell())
    check('mclc', bravais['mclc'](3.0, 4.0, 5.0, 75.0).tocell())
    check('tri', bravais['tri'](7.0, 6.0, 5.0, 65.0, 70.0, 80.0).tocell())
    check('sqr', bravais['sqr'](3.0).tocell())
    check('sqr', Cell(np.diag([3.0, 3.0, 10.0])), pbc=np.array([True, True, False]))
    check('crect', bravais['crect'](3.0, 40).tocell())
    alpha = 40 / 360 * 2 * np.pi
    a = 3
    x = np.cos(alpha)
    y = np.sin(alpha)
    crectcell = np.array([[a, 0, 0], [a * x, a * y, 0], [0, 0, 10]])
    check('crect', Cell(crectcell), pbc=[1, 1, 0])
    check('rect', bravais['rect'](3.0, 4.0).tocell())
    check('rect', Cell.new([3, 4, 10]), pbc=[1, 1, 0])
    check('hex2d', bravais['hex2d'](3.0).tocell())
    x = 0.5 * np.sqrt(3)
    hexcell = np.array([[a, 0, 0], [-0.5 * a, x * a, 0], [0.0, 0.0, 0.0]])
    check('hex2d', Cell(hexcell))
    check('obl', bravais['obl'](3.0, 4.0, 40).tocell())
    b = 4
    x = np.cos(alpha)
    y = np.sin(alpha)
    oblcell = np.array([[a, 0, 0], [b * x, b * y, 0], [0, 0, 10]])
    check('obl', Cell(oblcell), pbc=np.array([True, True, False]))
    check('line', Cell(np.diag([a, 0, 0.0])))
    check('line', Cell(np.diag([a, 1, 1.0])), pbc=np.array([1, 0, 0]))
    check('line', Cell(np.diag([0.0, 0, a])))
    check('line', Cell(np.diag([1.0, 1, a])), pbc=np.array([0, 0, 1]))