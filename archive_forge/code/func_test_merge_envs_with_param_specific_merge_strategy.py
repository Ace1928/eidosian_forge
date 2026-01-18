import json
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine import parameters
from heat.tests import common
from heat.tests import utils
def test_merge_envs_with_param_specific_merge_strategy(self):
    merge_strategies = {'default': 'overwrite', 'lst_value1': 'merge', 'lst_value2': 'merge', 'json_value1': 'deep_merge'}
    self.env_1['parameter_merge_strategies'] = merge_strategies
    files = {'env_1': json.dumps(self.env_1), 'env_2': json.dumps(self.env_2)}
    environment_files = ['env_1', 'env_2']
    env_util.merge_environments(environment_files, files, self.params, self.param_schemata)
    expected = {'parameters': {'json_value1': {u'3': [u'str3', u'str4'], u'1': [u'str1', u'str2']}, 'json_value2': {u'4': [u'test3', u'test4']}, 'del_lst_value1': '5,6', 'del_lst_value2': '7,8', 'lst_value1': [1, 2, 5, 6], 'str_value1': u'string3', 'str_value2': u'string4'}, 'resource_registry': {'test::R1': 'OS::Heat::RandomString', 'test::R2': 'OS::Heat::None'}, 'parameter_defaults': {'lst_value2': [3, 4, 7, 8]}}
    self.assertEqual(expected, self.params)