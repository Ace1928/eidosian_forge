import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def recordset_set(self, zone_id, id, record=None, type=None, description=None, ttl=None, no_description=False, no_ttl=False, *args, **kwargs):
    options_str = build_option_string({'--record': record, '--type': type, '--description': description, '--ttl': ttl})
    flags_str = build_flags_string({'--no-description': no_description, '--no-ttl': no_ttl})
    cmd = f'recordset set {zone_id} {id} {flags_str} {options_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)