import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def test_get_cinderclient_endpoint_exception(self):
    with mock.patch.object(cinder.ksa_session, 'Session'), mock.patch.object(cinder.ksa_identity, 'V3Password'), mock.patch.object(cinder.Store, 'is_user_overriden', return_value=False), mock.patch.object(cinder.keystone_sc, 'ServiceCatalogV2') as service_catalog:
        service_catalog.side_effect = keystone_exc.EndpointNotFound
        self.assertRaises(exceptions.BadStoreConfiguration, self.store.get_cinderclient, self.context)