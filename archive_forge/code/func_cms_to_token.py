import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def cms_to_token(cms_text):
    """Convert a CMS-signed token in PEM format to a custom URL-safe format.

    The conversion consists of replacing '/' char in the PEM-formatted token
    with the '-' char and doing other such textual replacements to make the
    result marshallable via HTTP. The return value can thus be used as the
    value of a HTTP header such as "X-Auth-Token".

    This ad-hoc conversion is an unfortunate oversight since the returned
    value now does not conform to any of the standard variants of base64
    encoding. It would have been better to use base64url encoding (either on
    the PEM formatted text or, perhaps even better, on the inner CMS-signed
    binary value without any PEM formatting). In any case, the same conversion
    is done in reverse in the other direction (for token verification), so
    there are no correctness issues here. Note that the non-standard encoding
    of the token will be preserved so as to not break backward compatibility.

    The conversion issue is detailed by the code author in a blog post at
    http://adam.younglogic.com/2014/02/compressed-tokens/.
    """
    start_delim = '-----BEGIN CMS-----'
    end_delim = '-----END CMS-----'
    signed_text = cms_text
    signed_text = signed_text.replace('/', '-')
    signed_text = signed_text.replace(start_delim, '')
    signed_text = signed_text.replace(end_delim, '')
    signed_text = signed_text.replace('\n', '')
    return signed_text