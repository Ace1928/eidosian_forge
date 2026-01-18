import copy
import json
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def get_strict_and_loose_templates(self, param_type):
    template_loose = template_format.parse(self.simple_template)
    template_loose['parameters']['param1']['type'] = param_type
    template_strict = copy.deepcopy(template_loose)
    template_strict['resources']['my_value']['properties']['type'] = param_type
    template_strict['resources']['my_value2']['properties']['type'] = param_type
    return (template_strict, template_loose)