from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking.routers import routers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.networking import resource_args
from googlecloudsdk.command_lib.edge_cloud.networking.routers import flags as routers_flags
from googlecloudsdk.core import log
def has_routes_arg(self, args):
    relevant_args = [args.add_advertisement_ranges, args.remove_advertisement_ranges, args.set_advertisement_ranges]
    filtered = filter(None, relevant_args)
    number_found = sum((1 for _ in filtered))
    if number_found == 0:
        return False
    if number_found == 1:
        return True
    raise ValueError('Invalid argument: Expected at most one of add_advertisement_ranges remove_advertisement_ranges set_advertisement_ranges')