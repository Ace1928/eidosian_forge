import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def header_max_count(self, name):
    """Return the maximum allowed number of headers named 'name'.

        Called when a header is added to a Message object.  If the returned
        value is not 0 or None, and there are already a number of headers with
        the name 'name' equal to the value returned, a ValueError is raised.

        Because the default behavior of Message's __setitem__ is to append the
        value to the list of headers, it is easy to create duplicate headers
        without realizing it.  This method allows certain headers to be limited
        in the number of instances of that header that may be added to a
        Message programmatically.  (The limit is not observed by the parser,
        which will faithfully produce as many headers as exist in the message
        being parsed.)

        The default implementation returns None for all header names.
        """
    return None