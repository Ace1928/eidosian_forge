import io
from urllib.request import urlopen
from urllib.error import HTTPError
def get_sprot_raw(id):
    """Get a text handle to a raw SwissProt entry at ExPASy.

    For an ID of XXX, fetches http://www.uniprot.org/uniprot/XXX.txt
    (as per the https://www.expasy.org/expasy_urls.html documentation).

    >>> from Bio import ExPASy
    >>> from Bio import SwissProt
    >>> with ExPASy.get_sprot_raw("O23729") as handle:
    ...     record = SwissProt.read(handle)
    ...
    >>> print(record.entry_name)
    CHS3_BROFI

    This function raises a ValueError if the identifier does not exist:

    >>> ExPASy.get_sprot_raw("DOES_NOT_EXIST")
    Traceback (most recent call last):
        ...
    ValueError: Failed to find SwissProt entry 'DOES_NOT_EXIST'

    """
    try:
        handle = _open(f'http://www.uniprot.org/uniprot/{id}.txt')
    except HTTPError as exception:
        if exception.code in (400, 404):
            raise ValueError(f"Failed to find SwissProt entry '{id}'") from None
        else:
            raise
    return handle