import numpy as np
from ase.units import Hartree, Bohr
def outstring(self):
    """Format yourself as a string"""
    string = '{0:g}  {1}  '.format(self.energy, self.index)

    def format_me(me):
        string = ''
        if me.dtype == float:
            for m in me:
                string += ' {0:g}'.format(m)
        else:
            for m in me:
                string += ' {0.real:g}{0.imag:+g}j'.format(m)
        return string
    string += '  ' + format_me(self.mur)
    if self.muv is not None:
        string += '  ' + format_me(self.muv)
    if self.magn is not None:
        string += '  ' + format_me(self.magn)
    string += '\n'
    return string