import base64
import datetime
from os import remove
from os.path import join
from OpenSSL import crypto
import dateutil.parser
import pytz
import saml2.cryptography.pki
def certificate_not_valid_yet(self, cert):
    starts_to_be_valid = dateutil.parser.parse(cert.get_notBefore())
    now = pytz.UTC.localize(datetime.datetime.utcnow())
    if starts_to_be_valid < now:
        return False
    return True