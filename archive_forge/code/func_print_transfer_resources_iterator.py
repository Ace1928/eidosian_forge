from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
def print_transfer_resources_iterator(resource_iterator, command_display_function, command_args):
    """Gcloud's built-in display logic has issues with enormous lists.

  Args:
    resource_iterator (iterable): Likely an instance of Apitools
      list_pager.YieldFromList but can also be a List.
    command_display_function (func): The self.Display function built into
      classes inheriting from base.Command.
    command_args (argparse.Namespace): The args object passed to self.Display
      and self.Run of commands inheriting from base.Command.
  """
    resource_list = []
    for resource in resource_iterator:
        resource_list.append(resource)
        if len(resource_list) >= _TRANSFER_LIST_PAGE_SIZE:
            log.status.Print()
            command_display_function(command_args, resource_list)
            resource_list = []
    if resource_list:
        log.status.Print()
        command_display_function(command_args, resource_list)
    command_args.format = None