import numpy as np
import os
import subprocess
import warnings
from ase.calculators.openmx.reader import rn as read_nth_to_last_value
def calc_dos(self, method='Tetrahedron', pdos=False, gaussian_width=0.1, atom_index_list=None):
    """
        Python interface for DosMain (OpenMX's density of states calculator).
        Can automate the density of states
        calculations used in OpenMX by processing .Dos.val and .Dos.vec files.
        :param method: method to be used to calculate the density of states
                       from eigenvalues and eigenvectors.
                       ('Tetrahedron' or 'Gaussian')
        :param pdos: If True, the pseudo-density of states is calculated for a
                     given list of atoms for each orbital. If the system is
                     spin polarized, then each up and down state is also
                     calculated.
        :param gaussian_width: If the method is 'Gaussian' then gaussian_width
                               is required (eV).
        :param atom_index_list: If pdos is True, a list of atom indices are
                                required to generate the pdos of each of those
                                specified atoms.
        :return: None
        """
    method_code = '2\n'
    if method == 'Tetrahedron':
        method_code = '1\n'
    pdos_code = '1\n'
    if pdos:
        pdos_code = '2\n'
    with open(os.path.join(self.calc.directory, 'std_dos.in'), 'w') as fd:
        fd.write(method_code)
        if method == 'Gaussian':
            fd.write(str(gaussian_width) + '\n')
        fd.write(pdos_code)
        if pdos:
            atoms_code = ''
            if atom_index_list is None:
                for i in range(len(self.calc.atoms)):
                    atoms_code += str(i + 1) + ' '
            else:
                for i in atom_index_list:
                    atoms_code += str(i) + ' '
            atoms_code += '\n'
            fd.write(atoms_code)
        fd.close()
    executable_name = 'DosMain'
    input_files = (self.calc.label + '.Dos.val', self.calc.label + '.Dos.vec', os.path.join(self.calc.directory, 'std_dos.in'))
    argument_format = '%s %s < %s'
    input_command(self.calc, executable_name, input_files, argument_format)