from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.core import log
def has_config_sync_error(fs):
    return fs and fs.configSyncState and fs.configSyncState.syncState and fs.configSyncState.syncState.errors