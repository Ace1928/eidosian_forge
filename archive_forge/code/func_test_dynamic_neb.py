import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT as OrigEMT
from ase.dyneb import DyNEB
from ase.optimize import BFGS
@pytest.mark.slow
def test_dynamic_neb():
    force_evaluations = [0]

    class EMT(OrigEMT):

        def calculate(self, *args, **kwargs):
            force_evaluations[0] += 1
            OrigEMT.calculate(self, *args, **kwargs)
    initial = fcc111('Pt', size=(3, 2, 3), orthogonal=True)
    initial.center(axis=2, vacuum=10)
    oxygen = Atoms('O')
    oxygen.translate(initial[7].position + (0.0, 0.0, 3.5))
    initial.extend(oxygen)
    initial.calc = EMT()
    opt = BFGS(initial)
    opt.run(fmax=0.03)
    final = initial.copy()
    final[18].x += 2.8
    final[18].y += 1.8
    final.calc = EMT()
    opt = BFGS(final)
    opt.run(fmax=0.03)
    images = [initial]
    for i in range(7):
        images.append(initial.copy())
    images.append(final)
    fmax = 0.03
    for i in range(1, len(images) - 1):
        calc = EMT()
        images[i].calc = calc

    def run_NEB():
        if method == 'dyn':
            neb = DyNEB(images, fmax=fmax, dynamic_relaxation=True)
            neb.interpolate()
        elif method == 'dyn_scale':
            neb = DyNEB(images, fmax=fmax, dynamic_relaxation=True, scale_fmax=6.0)
            neb.interpolate()
        else:
            neb = DyNEB(images, dynamic_relaxation=False)
            neb.interpolate()
        force_evaluations[0] = 0
        opt = BFGS(neb)
        opt.run(fmax=fmax)
        force_calls.append(force_evaluations[0])
        Emax.append(np.sort([image.get_potential_energy() for image in images[1:-1]])[-1])
    force_calls, Emax = ([], [])
    for method in ['def', 'dyn', 'dyn_scale']:
        run_NEB()
    print('\n# Force calls with default NEB: {}'.format(force_calls[0]))
    print('# Force calls with dynamic NEB: {}'.format(force_calls[1]))
    print('# Force calls with dynamic and scaled NEB: {}\n'.format(force_calls[2]))
    assert force_calls[2] < force_calls[1] < force_calls[0]
    assert abs(Emax[1] - Emax[0]) < 0.001
    assert abs(Emax[2] - Emax[0]) < 0.001