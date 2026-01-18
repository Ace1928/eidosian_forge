import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_dhcp_options(self, domaine_name: str=None, domaine_name_servers: list=None, ntp_servers: list=None, dry_run: bool=False):
    """
        Creates a new set of DHCP options, that you can then associate
        with a Net using the UpdateNet method.

        :param      domaine_name: Specify a domain name
        (for example, MyCompany.com). You can specify only one domain name.
        :type       domaine_name: ``str``

        :param      domaine_name_servers: The IP addresses of domain name
        servers. If no IP addresses are specified, the OutscaleProvidedDNS
        value is set by default.
        :type       domaine_name_servers: ``list`` of ``str``

        :param      ntp_servers: The IP addresses of the Network Time
        Protocol (NTP) servers.
        :type       ntp_servers: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The created Dhcp Options
        :rtype: ``dict``
        """
    action = 'CreateDhcpOptions'
    data = {'DryRun': dry_run}
    if domaine_name is not None:
        data.update({'DomaineName': domaine_name})
    if domaine_name_servers is not None:
        data.update({'DomaineNameServers': domaine_name_servers})
    if ntp_servers is not None:
        data.update({'NtpServers': ntp_servers})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['DhcpOptionsSet']
    return response.json()