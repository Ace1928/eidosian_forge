import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_instance_group_update_replace_huge_batch_size(self):
    """Test update replace with a huge batch size."""
    updt_template = self.ig_tmpl_with_updt_policy()
    group = updt_template['Resources']['JobServerGroup']
    policy = group['UpdatePolicy']['RollingUpdate']
    policy['MinInstancesInService'] = '0'
    policy['MaxBatchSize'] = '20'
    config = updt_template['Resources']['JobServerConfig']
    config['Properties']['UserData'] = 'new data'
    self.update_instance_group(updt_template, num_updates_expected_on_updt=5, num_creates_expected_on_updt=0, num_deletes_expected_on_updt=0, update_replace=True)