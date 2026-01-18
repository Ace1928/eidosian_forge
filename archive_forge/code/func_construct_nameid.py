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
def construct_nameid(self, userid, local_policy=None, sp_name_qualifier=None, name_id_policy=None, name_qualifier=''):
    """Returns a name_id for the userid. How the name_id is
        constructed depends on the context.

        :param local_policy: The policy the server is configured to follow
        :param userid: The local permanent identifier of the object
        :param sp_name_qualifier: The 'user'/-s of the name_id
        :param name_id_policy: The policy the server on the other side wants
            us to follow.
        :param name_qualifier: A domain qualifier
        :return: NameID instance precursor
        """
    args = self.nim_args(local_policy, sp_name_qualifier, name_id_policy)
    if name_qualifier:
        args['name_qualifier'] = name_qualifier
    else:
        args['name_qualifier'] = self.name_qualifier
    return self.get_nameid(userid, **args)