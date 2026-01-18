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
def assert_images_equal(images1, images2):
    assert len(images1) == len(images2), 'length mismatch'
    for atoms1, atoms2 in zip(images1, images2):
        differences = compare_atoms(atoms1, atoms2)
        assert not differences