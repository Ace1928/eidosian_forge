import copy
from hashlib import sha256
import logging
import shelve
from urllib.parse import quote
from urllib.parse import unquote
from saml2 import SAMLError
from saml2.s_utils import PolicyError
from saml2.s_utils import rndbytes
from saml2.saml import NAMEID_FORMAT_EMAILADDRESS
from saml2.saml import NAMEID_FORMAT_PERSISTENT
from saml2.saml import NAMEID_FORMAT_TRANSIENT
from saml2.saml import NameID
def handle_manage_name_id_request(self, name_id, new_id=None, new_encrypted_id='', terminate=''):
    """
        Requests from the SP is about the SPProvidedID attribute.
        So this is about adding,replacing and removing said attribute.

        :param name_id: NameID instance
        :param new_id: NewID instance
        :param new_encrypted_id: NewEncryptedID instance
        :param terminate: Terminate instance
        :return: The modified name_id
        """
    _id = self.find_local_id(name_id)
    orig_name_id = copy.copy(name_id)
    if new_id:
        name_id.sp_provided_id = new_id.text
    elif new_encrypted_id:
        pass
    elif terminate:
        name_id.sp_provided_id = None
    else:
        return name_id
    self.remove_remote(orig_name_id)
    self.store(_id, name_id)
    return name_id