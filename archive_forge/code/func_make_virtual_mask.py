import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.qmmm import combine_lj_lorenz_berthelot
from ase import units
import copy
def make_virtual_mask(self):
    virtual_mask = []
    ct1 = 0
    ct2 = 0
    for i in range(len(self.mask)):
        virtual_mask.append(self.mask[i])
        if self.mask[i]:
            ct1 += 1
        if not self.mask[i]:
            ct2 += 1
        if (ct2 == self.apm2) & (self.apm2 != self.atoms2.calc.sites_per_mol):
            virtual_mask.append(False)
            ct2 = 0
        if (ct1 == self.apm1) & (self.apm1 != self.atoms1.calc.sites_per_mol):
            virtual_mask.append(True)
            ct1 = 0
    self.virtual_mask = np.array(virtual_mask)