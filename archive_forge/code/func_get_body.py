import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def get_body(self, preferencelist=('related', 'html', 'plain')):
    """Return best candidate mime part for display as 'body' of message.

        Do a depth first search, starting with self, looking for the first part
        matching each of the items in preferencelist, and return the part
        corresponding to the first item that has a match, or None if no items
        have a match.  If 'related' is not included in preferencelist, consider
        the root part of any multipart/related encountered as a candidate
        match.  Ignore parts with 'Content-Disposition: attachment'.
        """
    best_prio = len(preferencelist)
    body = None
    for prio, part in self._find_body(self, preferencelist):
        if prio < best_prio:
            best_prio = prio
            body = part
            if prio == 0:
                break
    return body