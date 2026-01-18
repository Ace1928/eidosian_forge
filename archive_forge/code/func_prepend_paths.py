import base64
import logging
import os
import textwrap
import uuid
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import prettytable
from urllib import error
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient import exc
def prepend_paths(self, resource_path, stack_id):
    if stack_id not in self.id_to_res_info:
        return
    stack_id, res_name = self.id_to_res_info.get(stack_id)
    if res_name in self.id_to_res_info:
        n_stack_id, res_name = self.id_to_res_info.get(res_name)
        resource_path.insert(0, res_name)
        self.prepend_paths(resource_path, n_stack_id)
    elif res_name:
        resource_path.insert(0, res_name)