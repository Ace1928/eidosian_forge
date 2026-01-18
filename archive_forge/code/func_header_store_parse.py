import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def header_store_parse(self, name, value):
    """+
        The name and value are returned unmodified.
        """
    return (name, value)