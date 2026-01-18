from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def applicationsecuritygroup_to_dict(asg):
    return dict(id=asg.id, location=asg.location, name=asg.name, tags=asg.tags, provisioning_state=asg.provisioning_state)