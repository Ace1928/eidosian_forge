from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def pubchem_search(*args, mock_test=False, **kwargs):
    """
    Search PubChem for the field and search input on the argument passed in
    returning a PubchemData object. Note that only one argument may be passed
    in at a time.

    Parameters:
        name (str):
            the common name of the compound you're searching for
        cid (str or int):
            the cid of the compound you're searching for
        smiles (str):
            the smiles string of the compound you're searching for
        conformer (str or int):
            the conformer id of the compound you're searching for

    returns:
        result (PubchemData):
            a pubchem data object containing the information on the
            requested entry
    """
    search, field = analyze_input(*args, **kwargs)
    raw_pubchem = search_pubchem_raw(search, field, mock_test=mock_test)
    atoms, data = parse_pubchem_raw(raw_pubchem)
    result = PubchemData(atoms, data)
    return result