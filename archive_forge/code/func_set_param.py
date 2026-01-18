import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def set_param(self, param, value, header='Content-Type', requote=True, charset=None, language='', replace=False):
    """Set a parameter in the Content-Type header.

        If the parameter already exists in the header, its value will be
        replaced with the new value.

        If header is Content-Type and has not yet been defined for this
        message, it will be set to "text/plain" and the new parameter and
        value will be appended as per RFC 2045.

        An alternate header can be specified in the header argument, and all
        parameters will be quoted as necessary unless requote is False.

        If charset is specified, the parameter will be encoded according to RFC
        2231.  Optional language specifies the RFC 2231 language, defaulting
        to the empty string.  Both charset and language should be strings.
        """
    if not isinstance(value, tuple) and charset:
        value = (charset, language, value)
    if header not in self and header.lower() == 'content-type':
        ctype = 'text/plain'
    else:
        ctype = self.get(header)
    if not self.get_param(param, header=header):
        if not ctype:
            ctype = _formatparam(param, value, requote)
        else:
            ctype = SEMISPACE.join([ctype, _formatparam(param, value, requote)])
    else:
        ctype = ''
        for old_param, old_value in self.get_params(header=header, unquote=requote):
            append_param = ''
            if old_param.lower() == param.lower():
                append_param = _formatparam(param, value, requote)
            else:
                append_param = _formatparam(old_param, old_value, requote)
            if not ctype:
                ctype = append_param
            else:
                ctype = SEMISPACE.join([ctype, append_param])
    if ctype != self.get(header):
        if replace:
            self.replace_header(header, ctype)
        else:
            del self[header]
            self[header] = ctype