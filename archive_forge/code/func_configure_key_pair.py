import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def configure_key_pair(config):
    logger.info('Configuring keys for Ray Cluster Launcher to ssh into the head node.')
    assert os.path.exists(PRIVATE_KEY_PATH), 'Private key file at path {} was not found'.format(PRIVATE_KEY_PATH)
    assert os.path.exists(PUBLIC_KEY_PATH), 'Public key file at path {} was not found'.format(PUBLIC_KEY_PATH)
    config['auth']['ssh_private_key'] = PRIVATE_KEY_PATH
    public_key_remote_path = '~/{}'.format(PUBLIC_KEY_NAME_EXTN)
    config['file_mounts'][public_key_remote_path] = PUBLIC_KEY_PATH
    return config