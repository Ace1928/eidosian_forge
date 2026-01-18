import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def get_fake_nested_stack(self, names):
    nested_t = '\n        heat_template_version: 2015-04-30\n        description: Resource Group\n        resources:\n        '
    resource_snip = "\n        '%s':\n            type: SoftwareDeployment\n            properties:\n              foo: bar\n        "
    resources = [nested_t]
    for res_name in names:
        resources.extend([resource_snip % res_name])
    nested_t = ''.join(resources)
    return utils.parse_stack(template_format.parse(nested_t))