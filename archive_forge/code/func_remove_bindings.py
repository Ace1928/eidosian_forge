import copy
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.senlin import res_base
from heat.engine import translation
def remove_bindings(self, bindings):
    for bd in bindings:
        try:
            bd['action'] = self.client().detach_policy_from_cluster(bd[self.BD_CLUSTER], self.resource_id)['action']
            bd['finished'] = False
        except Exception as ex:
            if self.client_plugin().is_bad_request(ex) or self.client_plugin().is_not_found(ex):
                bd['finished'] = True
            else:
                raise