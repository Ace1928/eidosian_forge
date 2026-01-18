import copy
import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def register_store_opts(conf, reserved_stores=None):
    LOG.debug('Registering options for group %s', _STORE_CFG_GROUP)
    conf.register_opts(_STORE_OPTS, group=_STORE_CFG_GROUP)
    configured_backends = copy.deepcopy(conf.enabled_backends)
    if reserved_stores:
        conf.enabled_backends.update(reserved_stores)
        for key in reserved_stores.keys():
            fs_conf_template = [cfg.StrOpt('filesystem_store_datadir', default='/var/lib/glance/{}'.format(key), help=FS_CONF_DATADIR_HELP.format(key)), cfg.MultiStrOpt('filesystem_store_datadirs', help='Not used'), cfg.StrOpt('filesystem_store_metadata_file', help='Not used'), cfg.IntOpt('filesystem_store_file_perm', default=0, help='Not used'), cfg.IntOpt('filesystem_store_chunk_size', default=64 * units.Ki, min=1, help=FS_CONF_CHUNKSIZE_HELP.format(key)), cfg.BoolOpt('filesystem_thin_provisioning', default=False, help='Not used')]
            LOG.debug('Registering options for reserved store: {}'.format(key))
            conf.register_opts(fs_conf_template, group=key)
    driver_opts = _list_driver_opts()
    for backend in configured_backends:
        for opt_list in driver_opts:
            if configured_backends[backend] not in opt_list:
                continue
            LOG.debug('Registering options for group %s', backend)
            conf.register_opts(driver_opts[opt_list], group=backend)