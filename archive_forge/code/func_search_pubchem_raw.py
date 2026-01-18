from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def search_pubchem_raw(search, field, silent=False, mock_test=False):
    """
    A helper function for searching pubchem.

    Parameters:
        search (str or int):
            the compound you are searching for. This can be either
            a common name, CID, or smiles string depending of the
            `field` you are searching

        field (str):
            the particular field you are searching with. Possible values
            are 'name', 'CID', and 'smiles'.'name' will search common '
            'names,CID will search the Pubchem Chemical Idenitification '
            'Numberswhich can be found on their website and smiles'
            ' searches for compounds with the entered smiles string.

    returns:
        data (str):
            a string containing the raw response from pubchem.
    """
    suffix = 'sdf?record_type=3d'
    if field == 'conformers':
        url = '{}/{}/{}/{}'.format(base_url, field, str(search), suffix)
    else:
        url = '{}/compound/{}/{}/{}'.format(base_url, field, str(search), suffix)
    if mock_test:
        r = BytesIO(test_output)
    else:
        try:
            r = urllib.request.urlopen(url)
        except HTTPError as e:
            print(e.reason)
            raise ValueError('the search term {} could not be found for the field {}'.format(search, field))
        except URLError as e:
            print(e.reason)
            raise ValueError("Couldn't reach the pubchem servers, check your internet connection")
    if field != 'conformers' and (not silent):
        conformer_ids = available_conformer_search(search, field, mock_test=mock_test)
        if len(conformer_ids) > 1:
            warnings.warn('The structure "{}" has more than one conformer in PubChem. By default, the first conformer is returned, please ensure you are using the structure you intend to or use the `ase.data.pubchem.pubchem_conformer_search` function'.format(search))
    data = r.read().decode('utf-8')
    return data