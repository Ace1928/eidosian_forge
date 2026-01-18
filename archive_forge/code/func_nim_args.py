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
def nim_args(self, local_policy=None, sp_name_qualifier='', name_id_policy=None, name_qualifier=''):
    """

        :param local_policy:
        :param sp_name_qualifier:
        :param name_id_policy:
        :param name_qualifier:
        :return:
        """
    logger.debug('local_policy: %s, name_id_policy: %s', local_policy, name_id_policy)
    if name_id_policy and name_id_policy.sp_name_qualifier:
        sp_name_qualifier = name_id_policy.sp_name_qualifier
    else:
        sp_name_qualifier = sp_name_qualifier
    if name_id_policy and name_id_policy.format:
        nameid_format = name_id_policy.format
    elif local_policy:
        nameid_format = local_policy.get_nameid_format(sp_name_qualifier)
    else:
        raise SAMLError('Unknown NameID format')
    if not name_qualifier:
        name_qualifier = self.name_qualifier
    return {'nformat': nameid_format, 'sp_name_qualifier': sp_name_qualifier, 'name_qualifier': name_qualifier}