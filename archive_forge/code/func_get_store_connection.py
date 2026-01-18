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
def get_store_connection(self, auth_token, storage_url):
    """Get initialized swift connection

        :param auth_token: auth token
        :param storage_url: swift storage url
        :return: swiftclient connection that allows to request container and
                 others
        """
    return swiftclient.Connection(preauthurl=storage_url, preauthtoken=auth_token, insecure=self.insecure, ssl_compression=self.ssl_compression, cacert=self.cacert)