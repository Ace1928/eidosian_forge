import re
import numpy as np
from xml.dom import minidom
def get_ion(fname):
    """
    Read the ion.xml file of a specie
    Input parameters:
    -----------------
    fname (str): name of the ion file
    Output Parameters:
    ------------------
    ion (dict): The ion dictionary contains all the data
        from the ion file. Each field of the xml file give
        one key.
        The different keys are:
            'lmax_basis': int
            'self_energy': float
            'z': int
            'symbol': str
            'label': str
            'mass': flaot
            'lmax_projs': int
            'basis_specs': str
            'norbs_nl': int
            'valence': float
            'nprojs_nl: int

            The following keys give the pao field,
            'npts': list of int
            'delta':list of float
            'cutoff': list of float
            'data':list of np.arrayof shape (npts[i], 2)
            'orbital': list of dictionary
            'projector': list of dictionary

    """
    doc = minidom.parse(fname)
    elements_headers = [['symbol', str], ['label', str], ['z', int], ['valence', float], ['mass', float], ['self_energy', float], ['lmax_basis', int], ['norbs_nl', int], ['lmax_projs', int], ['nprojs_nl', int]]
    ion = {}
    for i, elname in enumerate(elements_headers):
        name = doc.getElementsByTagName(elname[0])
        ion[elname[0]] = get_data_elements(name[0], elname[1])
    name = doc.getElementsByTagName('basis_specs')
    ion['basis_specs'] = getNodeText(name[0])
    extract_pao_elements(ion, doc)
    return ion