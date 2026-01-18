from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def analyze_input(name=None, cid=None, smiles=None, conformer=None, silent=False):
    """
    helper function to translate keyword arguments from intialization
    and searching into the search and field that is being asked for

    Parameters:
        see `ase.data.pubchem.pubchem_search`
    returns:
        search:
            the search term the user has entered
        field:
            the name of the field being asked for

    """
    inputs = [name, cid, smiles, conformer]
    inputs_check = [a is not None for a in [name, cid, smiles, conformer]]
    input_fields = ['name', 'cid', 'smiles', 'conformers']
    if inputs_check.count(True) > 1:
        raise ValueError('Only one search term my be entered a time. Please pass in only one of the following: name, cid, smiles, confomer')
    elif inputs_check.count(True) == 1:
        index = inputs_check.index(True)
        field = input_fields[index]
        search = inputs[index]
    else:
        raise ValueError('No search was entered. Please pass in only one of the following: name, cid, smiles, confomer')
    return PubchemSearch(search, field)