import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def del_param(self, param, header='content-type', requote=True):
    """Remove the given parameter completely from the Content-Type header.

        The header will be re-written in place without the parameter or its
        value. All values will be quoted as necessary unless requote is
        False.  Optional header specifies an alternative to the Content-Type
        header.
        """
    if header not in self:
        return
    new_ctype = ''
    for p, v in self.get_params(header=header, unquote=requote):
        if p.lower() != param.lower():
            if not new_ctype:
                new_ctype = _formatparam(p, v, requote)
            else:
                new_ctype = SEMISPACE.join([new_ctype, _formatparam(p, v, requote)])
    if new_ctype != self.get(header):
        del self[header]
        self[header] = new_ctype