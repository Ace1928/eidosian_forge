import collections
import uuid
import weakref
from keystoneauth1 import exceptions as ks_exception
from keystoneauth1.identity import generic as ks_auth
from keystoneclient.v3 import client as kc_v3
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import importutils
from heat.common import config
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
@property
def domain_admin_auth(self):
    if not self._domain_admin_auth:
        auth = ks_auth.Password(username=self.domain_admin_user, password=self.domain_admin_password, auth_url=self.v3_endpoint, domain_id=self._stack_domain_id, domain_name=self.stack_domain_name, user_domain_id=self._stack_domain_id, user_domain_name=self.stack_domain_name)
        try:
            auth.get_token(self.session)
        except ks_exception.Unauthorized:
            LOG.error('Domain admin client authentication failed')
            raise exception.AuthorizationFailure()
        self._domain_admin_auth = auth
    return self._domain_admin_auth