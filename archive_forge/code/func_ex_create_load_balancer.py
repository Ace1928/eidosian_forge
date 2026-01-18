import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_load_balancer(self, load_balancer_name: str=None, load_balancer_type: str=None, security_groups: List[str]=None, subnets: List[str]=None, subregion_names: str=None, tag_keys: List[str]=None, tag_values: List[str]=None, l_backend_port: int=None, l_backend_protocol: str=None, l_load_balancer_port: int=None, l_load_balancer_protocol: str=None, l_server_certificate_id: str=None, dry_run: bool=False):
    """
        Creates a load balancer.
        The load balancer is created with a unique Domain Name Service (DNS)
        name. It receives the incoming traffic and routes it to its registered
        virtual machines (VMs). By default, this action creates an
        Internet-facing load balancer, resolving to public IP addresses.
        To create an internal load balancer in a Net, resolving to private IP
        addresses, use the LoadBalancerType parameter.

        :param      load_balancer_name: The name of the load balancer for
        which you want to create listeners. (required)
        :type       load_balancer_name: ``str``

        :param      load_balancer_type: The type of load balancer:
        internet-facing or internal. Use this parameter only for load
        balancers in a Net.
        :type       load_balancer_type: ``str``

        :param      security_groups: One or more IDs of security groups you
        want to assign to the load balancer.
        :type       security_groups: ``list`` of ``str``

        :param      subnets: One or more IDs of Subnets in your Net that you
        want to attach to the load balancer.
        :type       subnets: ``list`` of ``str``

        :param      subregion_names: One or more names of Subregions
        (currently, only one Subregion is supported). This parameter is not
        required if you create a load balancer in a Net. To create an internal
        load balancer, use the LoadBalancerType parameter.
        :type       subregion_names: ``list`` of ``str``

        :param      tag_keys: The key of the tag, with a minimum of 1
        character. (required)
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The value of the tag, between 0 and 255
        characters. (required)
        :type       tag_values: ``list`` of ``str``

        :param      l_backend_port: The port on which the back-end VM is
        listening (between 1 and 65535, both included). (required)
        :type       l_backend_port: ``int``

        :param      l_backend_protocol: The protocol for routing traffic to
        back-end VMs (HTTP | HTTPS | TCP | SSL | UDP).
        :type       l_backend_protocol: ``int``

        :param      l_load_balancer_port: The port on which the load balancer
        is listening (between 1 and 65535, both included). (required)
        :type       l_load_balancer_port: ``int``

        :param      l_load_balancer_protocol: The routing protocol
        (HTTP | HTTPS | TCP | SSL | UDP). (required)
        :type       l_load_balancer_protocol: ``str``

        :param      l_server_certificate_id: The ID of the server certificate.
        (required)
        :type       l_server_certificate_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Load Balancer
        :rtype: ``dict``
        """
    action = 'CreateLoadBalancer'
    data = {'DryRun': dry_run, 'Listeners': {}, 'Tags': {}}
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    if load_balancer_type is not None:
        data.update({'LoadBalencerType': load_balancer_type})
    if security_groups is not None:
        data.update({'SecurityGroups': security_groups})
    if subnets is not None:
        data.update({'Subnets': subnets})
    if subregion_names is not None:
        data.update({'SubregionNames': subregion_names})
    if tag_keys and tag_values and (len(tag_keys) == len(tag_values)):
        for key, value in zip(tag_keys, tag_values):
            data['Tags'].update({'Key': key, 'Value': value})
    if l_backend_port is not None:
        data['Listeners'].update({'BackendPort': l_backend_port})
    if l_backend_protocol is not None:
        data['Listeners'].update({'BackendProtocol': l_backend_protocol})
    if l_load_balancer_port is not None:
        data['Listeners'].update({'LoadBalancerPort': l_load_balancer_port})
    if l_load_balancer_protocol is not None:
        data['Listeners'].update({'LoadBalancerProtocol': l_load_balancer_protocol})
    if l_server_certificate_id is not None:
        data['Listeners'].update({'ServerCertificateId': l_server_certificate_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['LoadBalancer']
    return response.json()