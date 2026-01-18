import json
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine import parameters
from heat.tests import common
from heat.tests import utils
def test_merge_envs_with_param_defaults_conflicting_merge_strategy(self):
    merge_strategies = {'default': 'overwrite', 'lst_value2': 'merge'}
    env4_merge_strategies = {'default': 'overwrite', 'lst_value2': 'overwrite'}
    self.env_1['parameter_merge_strategies'] = merge_strategies
    self.env_4['parameter_merge_strategies'] = env4_merge_strategies
    files = {'env_1': json.dumps(self.env_1), 'env_2': json.dumps(self.env_2), 'env_4': json.dumps(self.env_4)}
    environment_files = ['env_1', 'env_2', 'env_4']
    self.assertRaises(exception.ConflictingMergeStrategyForParam, env_util.merge_environments, environment_files, files, self.params, self.param_schemata)