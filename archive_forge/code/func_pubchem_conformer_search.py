from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def pubchem_conformer_search(*args, mock_test=False, **kwargs):
    """
    Search PubChem for all the conformers of a given compound.
    Note that only one argument may be passed in at a time.

    Parameters:
        see `ase.data.pubchem.pubchem_search`

    returns:
        conformers (list):
            a list containing the PubchemData objects of all the conformers
            for your search
    """
    search, field = analyze_input(*args, **kwargs)
    conformer_ids = available_conformer_search(search, field, mock_test=mock_test)
    conformers = []
    for id_ in conformer_ids:
        conformers.append(pubchem_search(mock_test=mock_test, conformer=id_))
    return conformers