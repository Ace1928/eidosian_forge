import getpass
import hashlib
import sys
from keystoneauth1 import exceptions as ksa_exceptions
from oslo_utils import timeutils
from keystoneclient import exceptions as ksc_exceptions
def hash_signed_token(signed_text, mode='md5'):
    hash_ = hashlib.new(mode)
    hash_.update(signed_text)
    return hash_.hexdigest()