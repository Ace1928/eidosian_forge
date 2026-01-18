import numpy as np
from ase.units import Hartree, Bohr
def format_me(me):
    string = ''
    if me.dtype == float:
        for m in me:
            string += ' {0:g}'.format(m)
    else:
        for m in me:
            string += ' {0.real:g}{0.imag:+g}j'.format(m)
    return string