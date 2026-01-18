import numpy as np
import ase.units as un
def get_ground_state(self, atoms, **kw):
    """
        Run siesta calculations in order to get ground state properties.
        Makes sure that the proper parameters are unsed in order to be able
        to run pynao afterward, i.e.,

            COOP.Write = True
            WriteDenchar = True
            XML.Write = True
        """
    from ase.calculators.siesta import Siesta
    if 'fdf_arguments' not in kw.keys():
        kw['fdf_arguments'] = {'COOP.Write': True, 'WriteDenchar': True, 'XML.Write': True}
    else:
        for param in ['COOP.Write', 'WriteDenchar', 'XML.Write']:
            kw['fdf_arguments'][param] = True
    siesta = Siesta(**kw)
    atoms.calc = siesta
    atoms.get_potential_energy()