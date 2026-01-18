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
def get_nameid(self, userid, nformat, sp_name_qualifier, name_qualifier):
    if nformat == NAMEID_FORMAT_PERSISTENT:
        nameid = self.match_local_id(userid, sp_name_qualifier, name_qualifier)
        if nameid:
            logger.debug(f'Found existing persistent NameId {nameid} for user {userid}')
            return nameid
    _id = self.create_id(nformat, name_qualifier, sp_name_qualifier)
    if nformat == NAMEID_FORMAT_EMAILADDRESS:
        if not self.domain:
            raise SAMLError("Can't issue email nameids, unknown domain")
        _id = f'{_id}@{self.domain}'
    nameid = NameID(format=nformat, sp_name_qualifier=sp_name_qualifier, name_qualifier=name_qualifier, text=_id)
    self.store(userid, nameid)
    return nameid