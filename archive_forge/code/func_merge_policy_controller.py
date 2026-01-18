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
def merge_policy_controller(spec, config, version):
    """Merge configSync set in feature spec with the config template.

  ConfigSync has nested object structs need to be flatten.

  Args:
    spec: the ConfigManagementMembershipSpec message
    config: the dict loaded from full config template
    version: the version string of the membership
  """
    if not spec or not spec.policyController:
        return
    c = config[utils.POLICY_CONTROLLER]
    for field in list(config[utils.POLICY_CONTROLLER]):
        if hasattr(spec.policyController, field) and getattr(spec.policyController, field) is not None:
            c[field] = getattr(spec.policyController, field)
    valid_version = not version or semver.SemVer(version) >= semver.SemVer(utils.MONITORING_VERSION)
    spec_monitoring = spec.policyController.monitoring
    if not valid_version:
        c.pop('monitoring', None)
    elif spec_monitoring:
        c['monitoring'] = spec_monitoring