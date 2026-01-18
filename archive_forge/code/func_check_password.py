import itertools
from oslo_log import log
import passlib.hash
import keystone.conf
from keystone import exception
from keystone.i18n import _
def check_password(password, hashed):
    """Check that a plaintext password matches hashed.

    hashpw returns the salt value concatenated with the actual hash value.
    It extracts the actual salt if this value is then passed as the salt.

    """
    if password is None or hashed is None:
        return False
    password_utf8 = verify_length_and_trunc_password(password)
    hasher = _get_hasher_from_ident(hashed)
    return hasher.verify(password_utf8, hashed)