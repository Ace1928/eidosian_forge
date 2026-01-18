import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
def sniff(self, sample, delimiters=None):
    """
        Returns a dialect (or None) corresponding to the sample
        """
    quotechar, doublequote, delimiter, skipinitialspace = self._guess_quote_and_delimiter(sample, delimiters)
    if not delimiter:
        delimiter, skipinitialspace = self._guess_delimiter(sample, delimiters)
    if not delimiter:
        raise Error('Could not determine delimiter')

    class dialect(Dialect):
        _name = 'sniffed'
        lineterminator = '\r\n'
        quoting = QUOTE_MINIMAL
    dialect.doublequote = doublequote
    dialect.delimiter = delimiter
    dialect.quotechar = quotechar or '"'
    dialect.skipinitialspace = skipinitialspace
    return dialect