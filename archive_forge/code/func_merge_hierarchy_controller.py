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
def merge_hierarchy_controller(spec, config):
    if not spec or not spec.hierarchyController:
        return
    c = config[utils.HNC]
    for field in list(config[utils.HNC]):
        if hasattr(spec.hierarchyController, field) and getattr(spec.hierarchyController, field) is not None:
            c[field] = getattr(spec.hierarchyController, field)