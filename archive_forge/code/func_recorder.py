import numpy as np
from ase.build import bulk
from ase.units import fs
from ase.md import VelocityVerlet
from ase.md import Langevin
from ase.md import Andersen
from ase.io import Trajectory, read
import pytest
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
def recorder():
    temp.append(a_md.get_temperature())