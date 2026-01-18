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
def test_no_signal_action(self):
    self._create_stack(self.template)
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    rpcc = self.rpc_client
    rpcc.signal_software_deployment.return_value = 'deployment succeeded'
    details = {'foo': 'bar', 'deploy_status_code': 0}
    actions = [self.deployment.SUSPEND, self.deployment.DELETE]
    ev = self.patchobject(self.deployment, 'handle_signal')
    for action in actions:
        for status in self.deployment.STATUSES:
            self.deployment.state_set(action, status)
            self.deployment.signal(details)
            ev.assert_called_with(details)