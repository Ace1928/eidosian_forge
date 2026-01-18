from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import re
import pytest
from ase.cli.template import prec_round, slice_split, \
from ase.io import read
def test_cli_opt(cli, traj):
    stdout = cli.ase('diff', f'{traj}@:1', f'{traj}@:2', '-c', '--template', 'p1x,p2x,dx,f1x,f2x,dfx')
    stdout = stdout.split('\n')
    for counter, row in enumerate(stdout):
        if '=' in row:
            header = stdout[counter + 1]
            break
    header = re.sub('\\s+', ',', header).split(',')[1:-1]
    assert header == ['p1x', 'p2x', 'Δx', 'f1x', 'f2x', 'Δfx']
    cli.ase('diff', traj, '-c', '--template', 'p1x,f1x,p1y,f1y:0:-1,p1z,f1z,p1,f1', '--max-lines', '6', '--summary-functions', 'rmsd')