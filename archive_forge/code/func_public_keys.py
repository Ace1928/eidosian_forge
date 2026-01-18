import datetime
import os
import jwt
from oslo_utils import timeutils
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.token.providers import base
@property
def public_keys(self):
    keys = []
    key_repo = CONF.jwt_tokens.jws_public_key_repository
    for keyfile in os.listdir(key_repo):
        with open(os.path.join(key_repo, keyfile), 'r') as f:
            keys.append(f.read())
    return keys