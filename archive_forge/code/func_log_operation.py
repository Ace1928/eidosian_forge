from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def log_operation(resource_ref, action, past_tense, is_async=False):
    """Logs the long running operation of a resource.

  Args:
    resource_ref: A resource argument.
    action: str, present tense of the operation.
    past_tense: str, past tense of the operation.
    is_async: bool, if async operation is enabled.

  Returns:
    A string that logs the operation status.
  """
    resource_collection = resource_ref.Collection()
    resource_type = resource_collection.split('.')[-1]
    resource_type_to_name = {'vmwareClusters': 'user cluster in Anthos on VMware', 'vmwareNodePools': 'node pool of a user cluster in Anthos on VMware', 'vmwareAdminClusters': 'admin cluster in Anthos on VMware', 'bareMetalClusters': 'user cluster in Anthos on bare metal', 'bareMetalNodePools': 'node pool of a user cluster in Anthos on bare metal', 'bareMetalAdminClusters': 'admin cluster in Anthos on bare metal', 'bareMetalStandaloneClusters': 'standalone cluster in Anthos on bare metal', 'bareMetalStandaloneNodePools': 'node pool of a standalone cluster in Anthos on bare metal'}
    resource_name = resource_type_to_name.get(resource_type, 'unknown resource')
    self_link = resource_ref.SelfLink()
    if is_async:
        return '{action} in progress for {resource_name} [{self_link}].'.format(action=action, resource_name=resource_name, self_link=self_link)
    else:
        return '{past_tense} {resource_name} [{self_link}].'.format(past_tense=past_tense, resource_name=resource_name, self_link=self_link)