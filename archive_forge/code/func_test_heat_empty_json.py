from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_heat_empty_json(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {}, 'Parameters': {}, 'Outputs': {}}
    self._assert_can_create(tmpl)