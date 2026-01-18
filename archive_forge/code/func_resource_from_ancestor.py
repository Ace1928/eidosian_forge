from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import frozendict
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.asset import client_util as asset
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api as crm
from googlecloudsdk.command_lib.artifacts import requests as artifacts
def resource_from_ancestor(ancestor):
    """Converts an ancestor to a resource name.

  Args:
    ancestor: an ancestor proto return from GetAncestry

  Returns:
    The resource name of the ancestor
  """
    if ancestor.resourceId.type == 'organization':
        return 'organizations/{0}'.format(ancestor.resourceId.id)
    if ancestor.resourceId.type == 'folder':
        return 'folders/{0}'.format(ancestor.resourceId.id)
    if ancestor.resourceId.type == 'project':
        return 'projects/{0}'.format(ancestor.resourceId.id)