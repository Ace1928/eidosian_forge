from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bms.bms_client import BmsClient
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bms import flags
from googlecloudsdk.command_lib.bms import util
from googlecloudsdk.core.resource import resource_projector
def synthesizedInstance(self, instance, client):
    """Returns a synthesized Instance resource.

    Synthesized Instance has additional lists of networks for client and
    private.

    Args:
      instance: protorpc.messages.Message, The BMS instance.
      client: BmsClient, BMS API client.

    Returns:
      Synthesized Instance resource.

    """
    synthesized_instance = resource_projector.MakeSerializable(instance)
    client_networks = []
    private_networks = []
    for network in instance.networks:
        if client.IsClientNetwork(network):
            client_networks.append(network)
        elif client.IsPrivateNetwork(network):
            private_networks.append(network)
    if not client_networks and (not private_networks) and instance.logicalInterfaces:
        for logical_interface in instance.logicalInterfaces:
            for logical_network_interface in logical_interface.logicalNetworkInterfaces:
                if client.IsClientLogicalNetworkInterface(logical_network_interface):
                    client_networks.append(logical_network_interface)
                elif client.IsPrivateLogicalNetworkInterface(logical_network_interface):
                    private_networks.append(logical_network_interface)
    synthesized_instance['clientNetworks'] = client_networks
    synthesized_instance['privateNetworks'] = private_networks
    return synthesized_instance