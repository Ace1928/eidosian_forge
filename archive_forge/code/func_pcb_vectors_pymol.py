import warnings
from math import pi
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis
def pcb_vectors_pymol(self, filename='hs_exp.py'):
    """Write PyMol script for visualization.

        Write a PyMol script that visualizes the pseudo CB-CA directions
        at the CA coordinates.

        :param filename: the name of the pymol script file
        :type filename: string
        """
    if not self.ca_cb_list:
        warnings.warn('Nothing to draw.', RuntimeWarning)
        return
    with open(filename, 'w') as fp:
        fp.write('from pymol.cgo import *\n')
        fp.write('from pymol import cmd\n')
        fp.write('obj=[\n')
        fp.write('BEGIN, LINES,\n')
        fp.write(f'COLOR, {1.0:.2f}, {1.0:.2f}, {1.0:.2f},\n')
        for ca, cb in self.ca_cb_list:
            x, y, z = ca.get_array()
            fp.write(f'VERTEX, {x:.2f}, {y:.2f}, {z:.2f},\n')
            x, y, z = cb.get_array()
            fp.write(f'VERTEX, {x:.2f}, {y:.2f}, {z:.2f},\n')
        fp.write('END]\n')
        fp.write("cmd.load_cgo(obj, 'HS')\n")