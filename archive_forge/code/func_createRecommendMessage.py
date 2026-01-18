from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import network_troubleshooter
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute import user_permission_troubleshooter
from googlecloudsdk.command_lib.compute import vm_boot_troubleshooter
from googlecloudsdk.command_lib.compute import vm_status_troubleshooter
from googlecloudsdk.command_lib.compute import vpc_troubleshooter
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def createRecommendMessage(self, args, instance_name, instance_ref, project):
    release_track = ReleaseTrack.get(str(self.ReleaseTrack()).lower())
    release_track = release_track + ' ' if release_track else ''
    command = 'gcloud {0}compute ssh {1} --project={2} --zone={3} '.format(release_track, instance_name, project.name, args.zone or instance_ref.zone)
    if args.ssh_key_file:
        command += '--ssh-key-file={0} '.format(args.ssh_key_file)
    if args.force_key_file_overwrite:
        command += '--force-key-file-overwrite '
    command += '--troubleshoot'
    command_iap = command + ' --tunnel-through-iap'
    return RECOMMEND_MESSAGE.format(command, command_iap)