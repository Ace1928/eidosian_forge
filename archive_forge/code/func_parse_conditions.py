from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine.hot import template
from heat.engine import output
from heat.engine import properties
from heat.engine.resources.aws.autoscaling import autoscaling_group as aws_asg
from heat.engine import rsrc_defn
from heat.engine import support
def parse_conditions(self, stack, snippet, path=''):
    return snippet