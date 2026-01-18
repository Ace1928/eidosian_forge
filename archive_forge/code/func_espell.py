import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def espell(**keywds):
    """Retrieve spelling suggestions as a results handle.

    ESpell retrieves spelling suggestions, if available.

    See the online documentation for an explanation of the parameters:
    http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESpell

    Short example:

    >>> from Bio import Entrez
    >>> Entrez.email = "Your.Name.Here@example.org"
    >>> record = Entrez.read(Entrez.espell(term="biopythooon"))
    >>> print(record["Query"])
    biopythooon
    >>> print(record["CorrectedQuery"])
    biopython

    :returns: Handle to the results, by default in XML format.
    :raises urllib.error.URLError: If there's a network error.
    """
    cgi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi'
    variables = {}
    variables.update(keywds)
    request = _build_request(cgi, variables)
    return _open(request)