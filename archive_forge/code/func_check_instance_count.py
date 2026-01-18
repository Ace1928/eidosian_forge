import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def check_instance_count(self, stack_identifier, expected):
    md = self.client.resources.metadata(stack_identifier, 'custom_lb')
    actual_md = len(md['IPs'].split(','))
    if actual_md != expected:
        LOG.warning('check_instance_count exp:%d, meta:%s' % (expected, md['IPs']))
        return False
    stack = self.client.stacks.get(stack_identifier)
    inst_list = self._stack_output(stack, 'InstanceList')
    actual = len(inst_list.split(','))
    if actual != expected:
        LOG.warning('check_instance_count exp:%d, act:%s' % (expected, inst_list))
    return actual == expected