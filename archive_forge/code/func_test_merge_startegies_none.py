import json
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine import parameters
from heat.tests import common
from heat.tests import utils
def test_merge_startegies_none(self):
    merge_strategies = None
    param_strategy = env_util.get_param_merge_strategy(merge_strategies, 'param1')
    self.assertEqual(env_util.OVERWRITE, param_strategy)