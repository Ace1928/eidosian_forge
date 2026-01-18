import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.neb import NEB, NEBTools, idpp_interpolate
from ase.optimize import FIRE, BFGS
@pytest.mark.slow
def test_neb_tr(testdir):
    nimages = 3
    fmax = 0.01
    for remove_rotation_and_translation in [True, False]:
        initial = Atoms('O4', [(1.94366484, 2.24788196, 2.32204726), (3.05353823, 2.08091038, 2.30712548), (2.63770601, 3.05694348, 2.67368242), (2.50579418, 2.12540646, 3.28585811)])
        final = Atoms('O4', [(1.9550137, 2.22270649, 2.33191017), (3.07439495, 2.13662682, 2.31948449), (2.4473055, 1.26930465, 2.65964947), (2.52788189, 2.1899024, 3.29728667)])
        final.set_cell((5, 5, 5))
        initial.set_cell((5, 5, 5))
        final.calc = LennardJones()
        initial.calc = LennardJones()
        images = [initial]
        for i in range(nimages):
            image = initial.copy()
            image.calc = LennardJones()
            images.append(image)
        images.append(final)
        neb = NEB(images, remove_rotation_and_translation=remove_rotation_and_translation)
        neb.interpolate()
        idpp_interpolate(neb, fmax=0.1, optimizer=BFGS)
        qn = FIRE(neb, dt=0.005, maxstep=0.05, dtmax=0.1)
        qn.run(steps=20)
        neb = NEB(images, climb=True, remove_rotation_and_translation=remove_rotation_and_translation)
        qn = FIRE(neb, dt=0.005, maxstep=0.05, dtmax=0.1)
        qn.run(fmax=fmax)
        images = neb.images
        nebtools = NEBTools(images)
        Ef_neb, dE_neb = nebtools.get_barrier(fit=False)
        nsteps_neb = qn.nsteps
        if remove_rotation_and_translation:
            Ef_neb_0 = Ef_neb
            nsteps_neb_0 = nsteps_neb
    assert abs(Ef_neb - Ef_neb_0) < 0.01
    assert nsteps_neb_0 < nsteps_neb * 0.7