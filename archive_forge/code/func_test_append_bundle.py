import pytest
import numpy as np
import sys
from subprocess import check_call, check_output
from pathlib import Path
from ase.build import bulk
from ase.io import read, write
from ase.io.pickletrajectory import PickleTrajectory
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io.bundletrajectory import (BundleTrajectory,
@pytest.mark.xfail(reason='same as test_read_write_bundle')
def test_append_bundle(images, bundletraj):
    traj = BundleTrajectory(bundletraj, mode='a')
    assert len(read(bundletraj, ':')) == 2
    for atoms in images:
        traj.write(atoms)
    traj.close()
    images1 = read(bundletraj, ':')
    assert len(images1) == 4
    assert_images_equal(images * 2, images1)