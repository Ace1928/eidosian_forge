from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_cfn_empty_json(self):
    tmpl = {'AWSTemplateFormatVersion': '2010-09-09', 'Resources': {}, 'Parameters': {}, 'Outputs': {}}
    self._assert_can_create(tmpl)