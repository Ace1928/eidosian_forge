from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.container.fleet.features import info
def enable_poco_api_if_disabled(project):
    try:
        poco_api = info.Get('policycontroller').api
        enable_api.EnableServiceIfDisabled(project, poco_api)
    except:
        pass