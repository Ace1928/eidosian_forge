from barbicanclient import exceptions
from barbicanclient.v1 import client as barbican_client
from barbicanclient.v1 import containers
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_secret_by_ref(self, secret_ref):
    try:
        secret = self.client().secrets.get(secret_ref)
        secret.name
        return secret
    except Exception as ex:
        if self.is_not_found(ex):
            raise exception.EntityNotFound(entity='Secret', name=secret_ref)
        LOG.info('Failed to get Barbican secret from reference %s' % secret_ref)
        raise