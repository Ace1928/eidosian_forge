import io
from urllib.request import urlopen
import time
from Bio._utils import function_with_previous
def kegg_get(dbentries, option=None):
    """KEGG get - Data retrieval.

    dbentries - Identifiers (single string, or list of strings), see below.
    option - One of "aaseq", "ntseq", "mol", "kcf", "image", "kgml" (string)

    The input is limited up to 10 entries.
    The input is limited to one pathway entry with the image or kgml option.
    The input is limited to one compound/glycan/drug entry with the image option.

    Returns a handle.
    """
    if isinstance(dbentries, list) and len(dbentries) <= 10:
        dbentries = '+'.join(dbentries)
    elif isinstance(dbentries, list) and len(dbentries) > 10:
        raise ValueError('Maximum number of dbentries is 10 for kegg get query')
    if option in ['aaseq', 'ntseq', 'mol', 'kcf', 'image', 'kgml', 'json']:
        resp = _q('get', dbentries, option)
    elif option:
        raise ValueError('Invalid option arg for kegg get request.')
    else:
        resp = _q('get', dbentries)
    return resp