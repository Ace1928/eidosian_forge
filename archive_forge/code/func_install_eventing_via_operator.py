from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.events import exceptions
from googlecloudsdk.command_lib.events import stages
from googlecloudsdk.command_lib.events import util
from googlecloudsdk.command_lib.kuberun.core.events import init_shared
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
def install_eventing_via_operator(client, track):
    """Install eventing cluster by enabling it via the KubeRun operator.

  Attempt to determine whether KubeRun or CloudRun operator is installed by
    presence of the corresponding operator resource or namespace.

  Args:
    client: An api_tools client.
    track: base.ReleaseTrack, the release (ga, beta, alpha) the command is in.
  """
    namespaces_list = client.ListNamespaces()
    operator_obj = client.GetKubeRun()
    if operator_obj is not None and track == base.ReleaseTrack.ALPHA:
        operator_type = events_constants.Operator.KUBERUN
    elif 'cloud-run-system' in namespaces_list:
        operator_obj = client.GetCloudRun()
        operator_type = events_constants.Operator.CLOUDRUN
    else:
        operator_type = None
        init_shared.prompt_if_can_prompt('Unable to find the CloudRun resource to install Eventing. Eventing will not be installed. Would you like to continue anyway?')
        if 'cloud-run-events' in namespaces_list or 'events-system' in namespaces_list:
            log.status.Print('Eventing already installed.')
        else:
            raise exceptions.EventingInstallError('Eventing not installed.')
    if operator_obj is None:
        return
    tracker_stages = stages.EventingStages()
    operator_max_wait_secs = util.OPERATOR_MAX_WAIT_MS / 1000
    with progress_tracker.StagedProgressTracker('Waiting on eventing installation...' if operator_obj.eventing_enabled else 'Enabling eventing...', tracker_stages, failure_message='Eventing failed to install within {} seconds, please try rerunning the command.'.format(operator_max_wait_secs)) as tracker:
        if not operator_obj.eventing_enabled:
            _update_operator_with_eventing_enabled(client, operator_type)
        _poll_operator_resource(client, operator_type, tracker)
        if operator_obj.eventing_enabled:
            log.status.Print('Eventing already enabled.')
        else:
            log.status.Print('Enabled eventing successfully.')