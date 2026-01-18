from heat.common import template_format
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_dbinstance(self):
    """Test that Template is parsable and publishes correct properties."""
    templ = template.Template(template_format.parse(rds_template))
    stack = parser.Stack(utils.dummy_context(), 'test_stack', templ)
    res = stack['DatabaseServer']
    self.assertIsNone(res._validate_against_facade(DBInstance))