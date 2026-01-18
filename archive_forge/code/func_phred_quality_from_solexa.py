import warnings
from math import log
from Bio import BiopythonParserWarning
from Bio import BiopythonWarning
from Bio import StreamModeError
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from typing import (
def phred_quality_from_solexa(solexa_quality: float) -> float:
    """Convert a Solexa quality (which can be negative) to a PHRED quality.

    PHRED and Solexa quality scores are both log transformations of a
    probality of error (high score = low probability of error). This function
    takes a Solexa score, transforms it back to a probability of error, and
    then re-expresses it as a PHRED score. This assumes the error estimates
    are equivalent.

    The underlying formulas are given in the documentation for the sister
    function solexa_quality_from_phred, in this case the operation is::

        phred_quality = 10*log(10**(solexa_quality/10.0) + 1, 10)

    This will return a floating point number, it is up to you to round this to
    the nearest integer if appropriate.  e.g.

    >>> print("%0.2f" % round(phred_quality_from_solexa(80), 2))
    80.00
    >>> print("%0.2f" % round(phred_quality_from_solexa(20), 2))
    20.04
    >>> print("%0.2f" % round(phred_quality_from_solexa(10), 2))
    10.41
    >>> print("%0.2f" % round(phred_quality_from_solexa(0), 2))
    3.01
    >>> print("%0.2f" % round(phred_quality_from_solexa(-5), 2))
    1.19

    Note that a solexa_quality less then -5 is not expected, will trigger a
    warning, but will still be converted as per the logarithmic mapping
    (giving a number between 0 and 1.19 back).

    As a special case where None is used for a "missing value", None is
    returned:

    >>> print(phred_quality_from_solexa(None))
    None
    """
    if solexa_quality is None:
        return None
    if solexa_quality < -5:
        warnings.warn(f'Solexa quality less than -5 passed, {solexa_quality!r}', BiopythonWarning)
    return 10 * log(10 ** (solexa_quality / 10.0) + 1, 10)