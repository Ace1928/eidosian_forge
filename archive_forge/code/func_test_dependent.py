from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_dependent(self):
    r = gf.Flow('root')
    customer = test_utils.ProvidesRequiresTask('customer', provides=['dog'], requires=[])
    washer = test_utils.ProvidesRequiresTask('washer', requires=['dog'], provides=['wash'])
    dryer = test_utils.ProvidesRequiresTask('dryer', requires=['dog', 'wash'], provides=['dry_dog'])
    shaved = test_utils.ProvidesRequiresTask('shaver', requires=['dry_dog'], provides=['shaved_dog'])
    happy_customer = test_utils.ProvidesRequiresTask('happy_customer', requires=['shaved_dog'], provides=['happiness'])
    r.add(customer, washer, dryer, shaved, happy_customer)
    c = compiler.PatternCompiler(r).compile()
    self.assertEqual([], _get_scopes(c, customer))
    self.assertEqual([['washer', 'customer']], _get_scopes(c, dryer))
    self.assertEqual([['shaver', 'dryer', 'washer', 'customer']], _get_scopes(c, happy_customer))