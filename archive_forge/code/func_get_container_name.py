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
def get_container_name(self, image_id, default_image_container):
    """
        Returns appropriate container name depending upon value of
        ``swift_store_multiple_containers_seed``. In single-container mode,
        which is a seed value of 0, simply returns default_image_container.
        In multiple-container mode, returns default_image_container as the
        prefix plus a suffix determined by the multiple container seed

        examples:
            single-container mode:  'glance'
            multiple-container mode: 'glance_3a1' for image uuid 3A1xxxxxxx...

        :param image_id: UUID of image
        :param default_image_container: container name from
               ``swift_store_container``
        """
    if self.backend_group:
        seed_num_chars = getattr(self.conf, self.backend_group).swift_store_multiple_containers_seed
    else:
        seed_num_chars = self.conf.glance_store.swift_store_multiple_containers_seed
    if seed_num_chars is None or seed_num_chars < 0 or seed_num_chars > 32:
        reason = _('An integer value between 0 and 32 is required for swift_store_multiple_containers_seed.')
        LOG.error(reason)
        raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
    elif seed_num_chars > 0:
        image_id = str(image_id).lower()
        num_dashes = image_id[:seed_num_chars].count('-')
        num_chars = seed_num_chars + num_dashes
        name_suffix = image_id[:num_chars]
        new_container_name = default_image_container + '_' + name_suffix
        return new_container_name
    else:
        return default_image_container