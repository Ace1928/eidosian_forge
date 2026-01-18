from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def pubchem_atoms_conformer_search(*args, **kwargs):
    """
    Search PubChem for all the conformers of a given compound.
    Note that only one argument may be passed in at a time.

    Parameters:
        see `ase.data.pubchem.pubchem_search`

    returns:
        conformers (list):
            a list containing the atoms objects of all the conformers
            for your search
    """
    conformers = pubchem_conformer_search(*args, **kwargs)
    conformers = [conformer.get_atoms() for conformer in conformers]
    return conformers