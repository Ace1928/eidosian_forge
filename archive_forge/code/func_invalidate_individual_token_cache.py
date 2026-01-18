import base64
import datetime
import uuid
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.models import token_model
from keystone import notifications
def invalidate_individual_token_cache(self, token):
    self._validate_token.invalidate(self, token.id)
    token_values = self.revoke_api.model.build_token_values(token)
    self.check_revocation_v3.invalidate(self, token_values)