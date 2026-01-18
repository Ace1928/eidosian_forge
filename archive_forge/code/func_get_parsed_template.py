import os
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import stack_resource
def get_parsed_template(self):
    if cfg.CONF.loadbalancer_template:
        with open(cfg.CONF.loadbalancer_template) as templ_fd:
            LOG.info('Using custom loadbalancer template %s', cfg.CONF.loadbalancer_template)
            contents = templ_fd.read()
    else:
        contents = lb_template_default
    return template_format.parse(contents)