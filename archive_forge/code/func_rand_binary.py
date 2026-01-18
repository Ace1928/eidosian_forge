import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def rand_binary(x):
    return rng.randint(2, size=x)