from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import semver
def merge_config_sync(spec, config, version):
    """Merge configSync set in feature spec with the config template.

  ConfigSync has nested object structs need to be flatten.

  Args:
    spec: the ConfigManagementMembershipSpec message
    config: the dict loaded from full config template
    version: the version string of the membership
  """
    if not spec or not spec.configSync:
        return
    cs = config[utils.CONFIG_SYNC]
    git = spec.configSync.git
    oci = spec.configSync.oci
    if spec.configSync.enabled is not None:
        cs['enabled'] = spec.configSync.enabled
    elif git and git.syncRepo or (oci and oci.syncRepo):
        cs['enabled'] = True
    if spec.configSync.sourceFormat:
        cs['sourceFormat'] = spec.configSync.sourceFormat
    if not version or semver.SemVer(version) >= semver.SemVer(utils.PREVENT_DRIFT_VERSION):
        if spec.configSync.preventDrift:
            cs['preventDrift'] = spec.configSync.preventDrift
    else:
        del cs['preventDrift']
    if not git and (not oci):
        return
    if not version or semver.SemVer(version) >= semver.SemVer(utils.OCI_SUPPORT_VERSION):
        if git:
            cs['sourceType'] = 'git'
        elif oci:
            cs['sourceType'] = 'oci'
    else:
        del cs['sourceType']
    if cs['sourceType'] and cs['sourceType'] == 'oci':
        if oci.syncWaitSecs:
            cs['syncWait'] = oci.syncWaitSecs
        for field in ['policyDir', 'secretType', 'syncRepo', 'gcpServiceAccountEmail']:
            if hasattr(oci, field) and getattr(oci, field) is not None:
                cs[field] = getattr(oci, field)
    else:
        if git.syncWaitSecs:
            cs['syncWait'] = git.syncWaitSecs
        for field in ['policyDir', 'httpsProxy', 'secretType', 'syncBranch', 'syncRepo', 'syncRev', 'gcpServiceAccountEmail']:
            if hasattr(git, field) and getattr(git, field) is not None:
                cs[field] = getattr(git, field)