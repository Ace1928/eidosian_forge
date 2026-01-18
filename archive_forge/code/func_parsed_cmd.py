import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def parsed_cmd(self, cmd, model=None, *args, **kwargs):
    if self.using_auth_override:
        func = self._openstack_noauth
    else:
        func = self.openstack
    out = func(cmd, *args, **kwargs)
    LOG.debug(out)
    if model is not None:
        return model(out)
    return out