from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def test_hash_equal(self):
    rd1 = self.make_me_one_with_everything()
    rd2 = self.make_me_one_with_everything()
    self.assertEqual(rd1, rd2)
    self.assertEqual(hash(rd1), hash(rd2))