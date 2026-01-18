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
def iam_map(domain, project, skip_bucket, from_ar_permissions, best_effort=False):
    """Generates an AR-equivalent IAM mapping for a GCR registry.

  Args:
    domain: The domain of the GCR registry.
    project: The project of the GCR registry.
    skip_bucket: If true, get iam policy for project instead of bucket. This can
      be useful when the bucket doesn't exist.
    from_ar_permissions: If true, use AR permissions to generate roles that
      would not need to be added to AR since user already has equivalent access
      for docker commands
    best_effort: If true, lower the scope when encountering auth errors

  Returns:
    (map, failures) where map is a map of roles to sets of users and
    failures is a list of scopes that failed

  Raises:
    Exception: A problem was encountered while generating the policy.
  """
    if skip_bucket:
        resource = project_resource_name(project)
    else:
        resource = bucket_resource_name(domain, project)
    ancestry = crm.GetAncestry(project_id=project)
    failures = []
    analysis = None
    for num, ancestor in enumerate(reversed(ancestry.ancestor)):
        scope = resource_from_ancestor(ancestor)
        try:
            if from_ar_permissions:
                analysis = analyze_iam_policy(_AR_PERMISSIONS, resource, scope)
            else:
                analysis = analyze_iam_policy(_PERMISSIONS, resource, scope)
            break
        except apitools_exceptions.HttpForbiddenError:
            failures.append(scope)
            if not best_effort:
                raise
            if num == len(ancestry.ancestor) - 1:
                return (None, failures)
    if not analysis.fullyExplored or not analysis.mainAnalysis.fullyExplored:
        errors = list((err.cause for err in analysis.mainAnalysis.nonCriticalErrors))
        error_msg = '\n'.join(errors)
        raise ar_exceptions.ArtifactRegistryError(error_msg)
    perm_to_members = collections.defaultdict(set)
    for result in analysis.mainAnalysis.analysisResults:
        if not result.fullyExplored:
            raise ar_exceptions.ArtifactRegistryError(_ANALYSIS_NOT_FULLY_EXPLORED)
        if result.iamBinding.condition is not None and (not best_effort):
            raise ar_exceptions.ArtifactRegistryError('Conditional IAM binding is not supported.')
        members = set()
        for member in result.iamBinding.members:
            if is_convenience(member):
                continue
            members.add(member)
        for acl in result.accessControlLists:
            for access in acl.accesses:
                perm = access.permission
                perm_to_members[perm].update(members)
    role_to_members = collections.defaultdict(set)
    if from_ar_permissions:
        members = perm_to_members[_AR_PERMISSIONS_TO_ROLES[0][0]]
        for needed_perm, role in _AR_PERMISSIONS_TO_ROLES:
            members = members.intersection(perm_to_members[needed_perm])
            for member in members:
                role_to_members[role].add(member)
        return (role_to_members, failures)
    for perm, members in perm_to_members.items():
        role = _PERMISSION_TO_ROLE[perm]
        role_to_members[role].update(members)
    upgraded_members = set()
    final_map = collections.defaultdict(set)
    for role in _AR_ROLES:
        members = role_to_members[role]
        members.difference_update(upgraded_members)
        if not members:
            continue
        upgraded_members.update(members)
        final_map[role].update(members)
    return (final_map, failures)