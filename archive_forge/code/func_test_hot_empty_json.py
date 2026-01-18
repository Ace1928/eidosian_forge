from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_hot_empty_json(self):
    tmpl = {'heat_template_version': '2013-05-23', 'resources': {}, 'parameters': {}, 'outputs': {}}
    self._assert_can_create(tmpl)