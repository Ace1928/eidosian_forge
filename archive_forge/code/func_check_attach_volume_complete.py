from cinderclient import client as cc
from cinderclient import exceptions
from keystoneauth1 import exceptions as ks_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def check_attach_volume_complete(self, vol_id):
    vol = self.client().volumes.get(vol_id)
    if vol.status in ('available', 'attaching', 'reserved'):
        LOG.debug('Volume %(id)s is being attached - volume status: %(status)s', {'id': vol_id, 'status': vol.status})
        return False
    if vol.status != 'in-use':
        LOG.debug('Attachment failed - volume %(vol)s is in %(status)s status', {'vol': vol_id, 'status': vol.status})
        raise exception.ResourceUnknownStatus(resource_status=vol.status, result=_('Volume attachment failed'))
    LOG.info('Attaching volume %(id)s complete', {'id': vol_id})
    return True