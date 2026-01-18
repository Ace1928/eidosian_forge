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
def test_append_pickle(images, trajfile):
    with PickleTrajectory(trajfile, 'a', _warn=False) as traj:
        for image in images:
            traj.write(image)
    images1 = read_images(trajfile)
    assert_images_equal(images * 2, images1)