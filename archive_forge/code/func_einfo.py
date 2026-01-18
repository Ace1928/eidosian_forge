import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def einfo(**keywds):
    """Return a summary of the Entrez databases as a results handle.

    EInfo provides field names, index term counts, last update, and
    available links for each Entrez database.

    See the online documentation for an explanation of the parameters:
    http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EInfo

    Short example:

    >>> from Bio import Entrez
    >>> Entrez.email = "Your.Name.Here@example.org"
    >>> record = Entrez.read(Entrez.einfo())
    >>> 'pubmed' in record['DbList']
    True

    :returns: Handle to the results, by default in XML format.
    :raises urllib.error.URLError: If there's a network error.
    """
    cgi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi'
    variables = {}
    variables.update(keywds)
    request = _build_request(cgi, variables)
    return _open(request)