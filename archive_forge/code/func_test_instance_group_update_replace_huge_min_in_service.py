import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_instance_group_update_replace_huge_min_in_service(self):
    """Update replace with huge number of minimum instances in service."""
    updt_template = self.ig_tmpl_with_updt_policy()
    group = updt_template['Resources']['JobServerGroup']
    policy = group['UpdatePolicy']['RollingUpdate']
    policy['MinInstancesInService'] = '20'
    policy['MaxBatchSize'] = '2'
    policy['PauseTime'] = 'PT0S'
    config = updt_template['Resources']['JobServerConfig']
    config['Properties']['UserData'] = 'new data'
    self.update_instance_group(updt_template, num_updates_expected_on_updt=3, num_creates_expected_on_updt=2, num_deletes_expected_on_updt=2, update_replace=True)