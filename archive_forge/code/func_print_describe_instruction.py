from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workflows import cache
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def print_describe_instruction(response, args):
    """Prints describe execution command for just created execution of a workflow.

  Function to be used as a response hook
  (go/gcloud-declarative-commands#response)

  Args:
    response: API response
    args: gcloud command arguments

  Returns:
    response: API response
  """
    cmd_base = ' '.join(args.command_path[:-1])
    resource_name = six.text_type(response.name).split('/')
    execution_id = resource_name[-1]
    location = resource_name[3]
    log.status.Print('\nTo view the workflow status, you can use following command:')
    log.status.Print('{} executions describe {} --workflow {} --location {}'.format(cmd_base, execution_id, args.workflow, location))
    return response