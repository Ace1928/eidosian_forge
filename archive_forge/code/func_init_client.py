import http.client
import io
import logging
import math
import urllib.parse
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import identity as ks_identity
from keystoneauth1 import session as ks_session
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store.common import utils as gutils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI
from glance_store import location
def init_client(self, location, context=None):
    ref_params = sutils.SwiftParams(self.conf, backend=self.backend_group).params
    if self.backend_group:
        default_ref = getattr(self.conf, self.backend_group).default_swift_reference
    else:
        default_ref = self.conf.glance_store.default_swift_reference
    default_swift_reference = ref_params.get(default_ref)
    if not default_swift_reference:
        reason = (_('default_swift_reference %s is required.'), default_ref)
        LOG.error(reason)
        raise exceptions.BadStoreConfiguration(message=reason)
    auth_address = default_swift_reference.get('auth_address')
    user = default_swift_reference.get('user')
    key = default_swift_reference.get('key')
    user_domain_id = default_swift_reference.get('user_domain_id')
    user_domain_name = default_swift_reference.get('user_domain_name')
    project_domain_id = default_swift_reference.get('project_domain_id')
    project_domain_name = default_swift_reference.get('project_domain_name')
    if self.backend_group:
        self._set_url_prefix(context=context)
    trustor_auth = ks_identity.V3Token(auth_url=auth_address, token=context.auth_token, project_id=context.project_id)
    trustor_sess = ks_session.Session(auth=trustor_auth, verify=self.ks_verify)
    trustor_client = ks_client.Client(session=trustor_sess)
    auth_ref = trustor_client.session.auth.get_auth_ref(trustor_sess)
    roles = [t['name'] for t in auth_ref['roles']]
    tenant_name, user = user.split(':')
    password = ks_identity.V3Password(auth_url=auth_address, username=user, password=key, project_name=tenant_name, user_domain_id=user_domain_id, user_domain_name=user_domain_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
    trustee_sess = ks_session.Session(auth=password, verify=self.ks_verify)
    trustee_client = ks_client.Client(session=trustee_sess)
    trustee_user_id = trustee_client.session.get_user_id()
    trust_id = trustor_client.trusts.create(trustee_user=trustee_user_id, trustor_user=context.user_id, project=context.project_id, impersonation=True, role_names=roles).id
    client_password = ks_identity.V3Password(auth_url=auth_address, username=user, password=key, trust_id=trust_id, user_domain_id=user_domain_id, user_domain_name=user_domain_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
    client_sess = ks_session.Session(auth=client_password, verify=self.ks_verify)
    return ks_client.Client(session=client_sess)