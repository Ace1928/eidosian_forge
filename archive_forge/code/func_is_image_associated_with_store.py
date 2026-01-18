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
def is_image_associated_with_store(self, context, volume_id):
    """
        Updates legacy images URL to respective stores.
        This method checks the volume type of the volume associated with the
        image against the configured stores. It returns true if the
        cinder_volume_type configured in the store matches with the volume
        type of the image-volume. When cinder_volume_type is not configured
        then the it checks it against default_volume_type set in cinder.
        If above both conditions doesn't meet, it returns false.
        """
    try:
        cinder_client = self.get_cinderclient(context=context, legacy_update=True)
        cinder_volume_type = self.store_conf.cinder_volume_type
        volume = cinder_client.volumes.get(volume_id)
        if cinder_volume_type and volume.volume_type == cinder_volume_type:
            return True
        elif not cinder_volume_type:
            default_type = cinder_client.volume_types.default()
            if volume.volume_type == default_type.name:
                return True
    except Exception:
        pass
    return False