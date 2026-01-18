import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def esummary(**keywds):
    """Retrieve document summaries as a results handle.

    ESummary retrieves document summaries from a list of primary IDs or
    from the user's environment.

    See the online documentation for an explanation of the parameters:
    http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESummary

    This example discovers more about entry 19923 in the structure
    database:

    >>> from Bio import Entrez
    >>> Entrez.email = "Your.Name.Here@example.org"
    >>> handle = Entrez.esummary(db="structure", id="19923")
    >>> record = Entrez.read(handle)
    >>> handle.close()
    >>> print(record[0]["Id"])
    19923
    >>> print(record[0]["PdbDescr"])
    Crystal Structure Of E. Coli Aconitase B


    :returns: Handle to the results, by default in XML format.
    :raises urllib.error.URLError: If there's a network error.
    """
    cgi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
    variables = {}
    variables.update(keywds)
    request = _build_request(cgi, variables)
    return _open(request)