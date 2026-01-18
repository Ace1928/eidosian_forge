import contextlib
import errno
import importlib
import logging
import math
import os
import shlex
import socket
import time
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import exceptions as keystone_exc
from keystoneauth1 import identity as ksa_identity
from keystoneauth1 import session as ksa_session
from keystoneauth1 import token_endpoint as ksa_token_endpoint
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_utils import strutils
from oslo_utils import units
from glance_store._drivers.cinder import base
from glance_store import capabilities
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI, _LW
import glance_store.location
from the service catalog, and current context's user and project are used.
def get_cinderclient(self, context=None, legacy_update=False, version='3.0'):
    if legacy_update:
        user_overriden = False
        context = context.elevated()
    else:
        user_overriden = self.is_user_overriden()
    session = get_cinder_session(self.store_conf)
    if user_overriden:
        username = self.store_conf.cinder_store_user_name
        url = self.store_conf.cinder_store_auth_address
        auth = None
    else:
        username = context.user_id
        project = context.project_id
        token = context.auth_token or '%s:%s' % (username, project)
        if self.store_conf.cinder_endpoint_template:
            template = self.store_conf.cinder_endpoint_template
            url = template % context.to_dict()
        else:
            info = self.store_conf.cinder_catalog_info
            service_type, service_name, interface = info.split(':')
            try:
                catalog = keystone_sc.ServiceCatalogV2(context.service_catalog)
                url = catalog.url_for(region_name=self.store_conf.cinder_os_region_name, service_type=service_type, service_name=service_name, interface=interface)
            except keystone_exc.EndpointNotFound:
                reason = _('Failed to find Cinder from a service catalog.')
                raise exceptions.BadStoreConfiguration(store_name='cinder', reason=reason)
        auth = ksa_token_endpoint.Token(endpoint=url, token=token)
    api_version = api_versions.APIVersion(version)
    c = cinderclient.Client(session=session, auth=auth, region_name=self.store_conf.cinder_os_region_name, retries=self.store_conf.cinder_http_retries, api_version=api_version)
    LOG.debug('Cinderclient connection created for user %(user)s using URL: %(url)s.', {'user': username, 'url': url})
    return c