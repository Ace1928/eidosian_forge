from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def pubchem_atoms_search(*args, **kwargs):
    """
    Search PubChem for the field and search input on the argument passed in
    returning an atoms object.Note that only one argument may be passed
    in at a time.

    Parameters:
        see `ase.data.pubchem.pubchem_search`

    returns:
        atoms (ASE Atoms Object):
            an ASE Atoms object containing the information on the
            requested entry
    """
    return pubchem_search(*args, **kwargs).get_atoms()