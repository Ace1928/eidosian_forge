import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_instance_group_update_no_replace_with_adjusted_capacity(self):
    """Test update only and no replace with capacity adjustment.

        Test update only and no replace (i.e. updated instance flavor in
        Launch Configuration) with capacity adjustment due to conflict in
        batch size and minimum instances in service.
        """
    updt_template = self.ig_tmpl_with_updt_policy()
    group = updt_template['Resources']['JobServerGroup']
    policy = group['UpdatePolicy']['RollingUpdate']
    policy['MinInstancesInService'] = '4'
    policy['MaxBatchSize'] = '4'
    policy['PauseTime'] = 'PT0S'
    config = updt_template['Resources']['JobServerConfig']
    config['Properties']['InstanceType'] = self.conf.instance_type
    self.update_instance_group(updt_template, num_updates_expected_on_updt=2, num_creates_expected_on_updt=3, num_deletes_expected_on_updt=3, update_replace=False)