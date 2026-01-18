import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def check_resources():

    def is_complete(r):
        return r.resource_status in {'CREATE_COMPLETE', 'UPDATE_COMPLETE'}
    resources = self.list_resources(stack_identifier, is_complete)
    if len(resources) < 2:
        return False
    self.assertIn('test3', resources)
    return True