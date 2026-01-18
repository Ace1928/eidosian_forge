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
def get_cinder_session(conf):
    global CINDER_SESSION
    if not CINDER_SESSION:
        auth = ksa_identity.V3Password(password=conf.cinder_store_password, username=conf.cinder_store_user_name, user_domain_name=conf.cinder_store_user_domain_name, project_name=conf.cinder_store_project_name, project_domain_name=conf.cinder_store_project_domain_name, auth_url=conf.cinder_store_auth_address)
        if conf.cinder_api_insecure:
            verify = False
        elif conf.cinder_ca_certificates_file:
            verify = conf.cinder_ca_certificates_file
        else:
            verify = True
        CINDER_SESSION = ksa_session.Session(auth=auth, verify=verify)
    return CINDER_SESSION