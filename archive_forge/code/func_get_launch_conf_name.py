import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def get_launch_conf_name(self, stack, ig_name):
    return stack[ig_name].properties['LaunchConfigurationName']