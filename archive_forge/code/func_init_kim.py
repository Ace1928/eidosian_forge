import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
def init_kim(self):
    """Create the KIM API Portable Model object and KIM API ComputeArguments
        object
        """
    if self.kim_initialized:
        return
    self.kim_model = kimpy_wrappers.PortableModel(self.model_name, self.debug)
    self.compute_args = self.kim_model.compute_arguments_create()