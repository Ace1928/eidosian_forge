import datetime
from hashlib import sha1
import logging
from pymongo import MongoClient
import pymongo.errors
import pymongo.uri_parser
from saml2.eptid import Eptid
from saml2.ident import IdentDB
from saml2.ident import Unknown
from saml2.ident import code_binary
from saml2.mdie import from_dict
from saml2.mdie import to_dict
from saml2.mdstore import InMemoryMetaData
from saml2.mdstore import load_metadata_modules
from saml2.mdstore import metadata_modules
from saml2.s_utils import PolicyError
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def get_assertion(self, cid):
    res = []
    for item in self.assertion.find({'assertion_id': cid}):
        res.append({'assertion': from_dict(item['assertion'], ONTS, True), 'to_sign': item['to_sign']})
    if len(res) == 1:
        return res[0]
    elif res is []:
        return None
    else:
        raise SystemError('More then one assertion with the same ID')