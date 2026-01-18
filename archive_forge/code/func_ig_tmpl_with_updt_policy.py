import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def ig_tmpl_with_updt_policy(self):
    templ = json.loads(copy.deepcopy(self.template))
    up = {'RollingUpdate': {'MinInstancesInService': '1', 'MaxBatchSize': '2', 'PauseTime': 'PT1S'}}
    templ['Resources']['JobServerGroup']['UpdatePolicy'] = up
    return templ