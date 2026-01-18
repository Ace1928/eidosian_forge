import io
from urllib.request import urlopen
import time
from Bio._utils import function_with_previous
def kegg_find(database, query, option=None):
    """KEGG find - Data search.

    Finds entries with matching query keywords or other query data in
    a given database.

    db - database or organism (string)
    query - search terms (string)
    option - search option (string), see below.

    For the compound and drug database, set option to the string 'formula',
    'exact_mass' or 'mol_weight' to search on that field only. The
    chemical formula search is a partial match irrespective of the order
    of atoms given. The exact mass (or molecular weight) is checked by
    rounding off to the same decimal place as the query data. A range of
    values may also be specified with the minus(-) sign.

    """
    if database in ['compound', 'drug'] and option in ['formula', 'exact_mass', 'mol_weight']:
        resp = _q('find', database, query, option)
    elif option:
        raise ValueError('Invalid option arg for kegg find request.')
    else:
        if isinstance(query, list):
            query = '+'.join(query)
        resp = _q('find', database, query)
    return resp