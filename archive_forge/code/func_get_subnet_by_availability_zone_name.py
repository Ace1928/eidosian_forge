import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def get_subnet_by_availability_zone_name(client, share_network_id, az_name):
    subnets = client.get_share_network_subnets(share_network_id)
    return next((subnet for subnet in subnets if subnet['availability_zone'] == az_name), None)