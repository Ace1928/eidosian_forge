import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def shared_zone_list(self, zone_id, *args, **kwargs):
    cmd = f'zone share list {zone_id}'
    return self.parsed_cmd(cmd, ListModel, *args, **kwargs)