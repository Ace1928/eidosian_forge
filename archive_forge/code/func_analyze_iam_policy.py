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
def analyze_iam_policy(permissions, resource, scope):
    """Calls AnalyzeIamPolicy for the given resource.

  Args:
    permissions: for the access selector
    resource: for the resource selector
    scope: for the scope

  Returns:
    An CloudassetAnalyzeIamPolicyResponse.
  """
    client = asset.GetClient()
    service = client.v1
    messages = asset.GetMessages()
    encoding.AddCustomJsonFieldMapping(messages.CloudassetAnalyzeIamPolicyRequest, 'analysisQuery_resourceSelector_fullResourceName', 'analysisQuery.resourceSelector.fullResourceName')
    encoding.AddCustomJsonFieldMapping(messages.CloudassetAnalyzeIamPolicyRequest, 'analysisQuery_accessSelector_permissions', 'analysisQuery.accessSelector.permissions')
    return service.AnalyzeIamPolicy(messages.CloudassetAnalyzeIamPolicyRequest(analysisQuery_accessSelector_permissions=permissions, analysisQuery_resourceSelector_fullResourceName=resource, scope=scope))