import datetime
from pyasn1 import error
from pyasn1.compat import dateandtime
from pyasn1.compat import string
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
@classmethod
def fromDateTime(cls, dt):
    """Create |ASN.1| object from a :py:class:`datetime.datetime` object.

        Parameters
        ----------
        dt: :py:class:`datetime.datetime` object
            The `datetime.datetime` object to initialize the |ASN.1| object
            from

        Returns
        -------
        :
            new instance of |ASN.1| value
        """
    text = dt.strftime(cls._yearsDigits == 4 and '%Y%m%d%H%M%S' or '%y%m%d%H%M%S')
    if cls._hasSubsecond:
        text += '.%d' % (dt.microsecond // 10000)
    if dt.utcoffset():
        seconds = dt.utcoffset().seconds
        if seconds < 0:
            text += '-'
        else:
            text += '+'
        text += '%.2d%.2d' % (seconds // 3600, seconds % 3600)
    else:
        text += 'Z'
    return cls(text)