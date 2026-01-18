import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_node_uncustomized(self, name, image, ex_network_domain, ex_is_started=True, ex_description=None, ex_cluster_id=None, ex_cpu_specification=None, ex_memory_gb=None, ex_primary_nic_private_ipv4=None, ex_primary_nic_vlan=None, ex_primary_nic_network_adapter=None, ex_additional_nics=None, ex_disks=None, ex_tagid_value_pairs=None, ex_tagname_value_pairs=None):
    """
        This MCP 2.0 only function deploys a new Cloud Server from a
        CloudControl compatible Server Image, which does not utilize
        VMware Guest OS Customization process.

        Create Node in MCP2 Data Center

        :keyword    name:   (required) String with a name for this new node
        :type       name:   ``str``

        :keyword    image:  (UUID of the Server Image being used as the target
                            for the new Server deployment. The source Server
                            Image (OS Image or Customer Image) must have
                            osCustomization set to true. See Get/List OS
                            Image(s) and Get/List Customer Image(s).
        :type       image:  :class:`NodeImage` or ``str``


        :keyword    ex_network_domain:  (required) Network Domain or Network
                                        Domain ID to create the node
        :type       ex_network_domain: :class:`NttCisNetworkDomain`
                                        or ``str``

        :keyword    ex_description:  (optional) description for this node
        :type       ex_description:  ``str``

        :keyword    ex_cluster_id:  (optional) For multiple cluster
        environments, it is possible to set a destination cluster for the new
        Customer Image. Note that performance of this function is optimal when
        either the Server cluster and destination are the same or when shared
        data storage is in place for the multiple clusters.
        :type       ex_cluster_id:  ``str``


        :keyword    ex_primary_nic_private_ipv4:  Provide private IPv4. Ignore
                                                  if ex_primary_nic_vlan is
                                                  provided. Use one or the
                                                  other. Not both.
        :type       ex_primary_nic_private_ipv4: :``str``

        :keyword    ex_primary_nic_vlan:  Provide VLAN for the node if
                                          ex_primary_nic_private_ipv4 NOT
                                          provided. One or the other. Not both.
        :type       ex_primary_nic_vlan: :class: NttCisVlan or ``str``

        :keyword    ex_primary_nic_network_adapter: (Optional) Default value
                                                    for the Operating System
                                                    will be used if leave
                                                    empty. Example: "E1000".
        :type       ex_primary_nic_network_adapter: :``str``

        :keyword    ex_additional_nics: (optional) List
                                        :class:'NttCisNic' or None
        :type       ex_additional_nics: ``list`` of :class:'NttCisNic'
                                        or ``str``

        :keyword    ex_memory_gb:  (optional) The amount of memory in GB for
                                   the server Can be used to override the
                                   memory value inherited from the source
                                   Server Image.
        :type       ex_memory_gb: ``int``

        :keyword    ex_cpu_specification: (optional) The spec of CPU to deploy
        :type       ex_cpu_specification:
                        :class:`NttCisServerCpuSpecification`

        :keyword    ex_is_started: (required) Start server after creation.
                                   Default is set to true.
        :type       ex_is_started:  ``bool``

        :keyword    ex_disks: (optional) NttCis disks. Optional disk
                            elements can be used to define the disk speed
                            that each disk on the Server; inherited from the
                            source Server Image will be deployed to. It is
                            not necessary to include a diskelement for every
                            disk; only those that you wish to set a disk
                            speed value for. Note that scsiId 7 cannot be
                            used.Up to 13 disks can be present in addition to
                            the required OS disk on SCSI ID 0. Refer to
                            https://docs.mcp-services.net/x/UwIu for disk

        :type       ex_disks: List or tuple of :class:'NttCisServerDisk`

        :keyword    ex_tagid_value_pairs:
                            (Optional) up to 10 tag elements may be provided.
                            A combination of tagById and tag name cannot be
                            supplied in the same request.
                            Note: ex_tagid_value_pairs and
                            ex_tagname_value_pairs is
                            mutually exclusive. Use one or other.

        :type       ex_tagname_value_pairs: ``dict``.  Value can be None.

        :keyword    ex_tagname_value_pairs:
                            (Optional) up to 10 tag elements may be provided.
                            A combination of tagById and tag name cannot be
                            supplied in the same request.
                            Note: ex_tagid_value_pairs and
                            ex_tagname_value_pairs is
                            mutually exclusive. Use one or other.

        :type       ex_tagname_value_pairs: ``dict```.

        :return: The newly created :class:`Node`.
        :rtype: :class:`Node`
        """
    if LooseVersion(self.connection.active_api_version) < LooseVersion('2.4'):
        raise Exception('This feature is NOT supported in  earlier api version of 2.4')
    if not isinstance(ex_is_started, bool):
        ex_is_started = True
        print('Warning: ex_is_started input value is invalid. Defaultto True')
    server_uncustomized_elm = ET.Element('deployUncustomizedServer', {'xmlns': TYPES_URN})
    ET.SubElement(server_uncustomized_elm, 'name').text = name
    ET.SubElement(server_uncustomized_elm, 'description').text = ex_description
    image_id = self._image_to_image_id(image)
    ET.SubElement(server_uncustomized_elm, 'imageId').text = image_id
    if ex_cluster_id:
        dns_elm = ET.SubElement(server_uncustomized_elm, 'primaryDns')
        dns_elm.text = ex_cluster_id
    if ex_is_started is not None:
        ET.SubElement(server_uncustomized_elm, 'start').text = str(ex_is_started).lower()
    if ex_cpu_specification is not None:
        cpu = ET.SubElement(server_uncustomized_elm, 'cpu')
        cpu.set('speed', ex_cpu_specification.performance)
        cpu.set('count', str(ex_cpu_specification.cpu_count))
        cpu.set('coresPerSocket', str(ex_cpu_specification.cores_per_socket))
    if ex_memory_gb is not None:
        ET.SubElement(server_uncustomized_elm, 'memoryGb').text = str(ex_memory_gb)
    if ex_primary_nic_private_ipv4 is None and ex_primary_nic_vlan is None:
        raise ValueError('Missing argument. Either ex_primary_nic_private_ipv4 or ex_primary_nic_vlan must be specified.')
    if ex_primary_nic_private_ipv4 is not None and ex_primary_nic_vlan is not None:
        raise ValueError('Either ex_primary_nic_private_ipv4 or ex_primary_nic_vlan be specified. Not both.')
    network_elm = ET.SubElement(server_uncustomized_elm, 'networkInfo')
    net_domain_id = self._network_domain_to_network_domain_id(ex_network_domain)
    network_elm.set('networkDomainId', net_domain_id)
    pri_nic = ET.SubElement(network_elm, 'primaryNic')
    if ex_primary_nic_private_ipv4 is not None:
        ET.SubElement(pri_nic, 'privateIpv4').text = ex_primary_nic_private_ipv4
    if ex_primary_nic_vlan is not None:
        vlan_id = self._vlan_to_vlan_id(ex_primary_nic_vlan)
        ET.SubElement(pri_nic, 'vlanId').text = vlan_id
    if ex_primary_nic_network_adapter is not None:
        ET.SubElement(pri_nic, 'networkAdapter').text = ex_primary_nic_network_adapter
    if isinstance(ex_additional_nics, (list, tuple)):
        for nic in ex_additional_nics:
            additional_nic = ET.SubElement(network_elm, 'additionalNic')
            if nic.private_ip_v4 is None and nic.vlan is None:
                raise ValueError('Either a vlan or private_ip_v4 must be specified for each additional nic.')
            if nic.private_ip_v4 is not None and nic.vlan is not None:
                raise ValueError('Either a vlan or private_ip_v4 must be specified for each additional nic. Not both.')
            if nic.private_ip_v4 is not None:
                ET.SubElement(additional_nic, 'privateIpv4').text = nic.private_ip_v4
            if nic.vlan is not None:
                vlan_id = self._vlan_to_vlan_id(nic.vlan)
                ET.SubElement(additional_nic, 'vlanId').text = vlan_id
            if nic.network_adapter_name is not None:
                ET.SubElement(additional_nic, 'networkAdapter').text = nic.network_adapter_name
    elif ex_additional_nics is not None:
        raise TypeError('ex_additional_NICs must be None or tuple/list')
    if isinstance(ex_disks, (list, tuple)):
        for disk in ex_disks:
            disk_elm = ET.SubElement(server_uncustomized_elm, 'disk')
            disk_elm.set('scsiId', disk.scsi_id)
            disk_elm.set('speed', disk.speed)
    elif ex_disks is not None:
        raise TypeError('ex_disks must be None or tuple/list')
    if ex_tagid_value_pairs is not None and ex_tagname_value_pairs is not None:
        raise ValueError('ex_tagid_value_pairs and ex_tagname_value_pairsis mutually exclusive. Use one or the other.')
    if ex_tagid_value_pairs is not None:
        if not isinstance(ex_tagid_value_pairs, dict):
            raise ValueError('ex_tagid_value_pairs must be a dictionary.')
        if sys.version_info[0] < 3:
            tagid_items = ex_tagid_value_pairs.iteritems()
        else:
            tagid_items = ex_tagid_value_pairs.items()
        for k, v in tagid_items:
            tag_elem = ET.SubElement(server_uncustomized_elm, 'tagById')
            ET.SubElement(tag_elem, 'tagKeyId').text = k
            if v is not None:
                ET.SubElement(tag_elem, 'value').text = v
    if ex_tagname_value_pairs is not None:
        if not isinstance(ex_tagname_value_pairs, dict):
            raise ValueError('ex_tagname_value_pairs must be a dictionary')
        if sys.version_info[0] < 3:
            tags_items = ex_tagname_value_pairs.iteritems()
        else:
            tags_items = ex_tagname_value_pairs.items()
        for k, v in tags_items:
            tag_name_elem = ET.SubElement(server_uncustomized_elm, 'tag')
            ET.SubElement(tag_name_elem, 'tagKeyName').text = k
            if v is not None:
                ET.SubElement(tag_name_elem, 'value').text = v
    response = self.connection.request_with_orgId_api_2('server/deployUncustomizedServer', method='POST', data=ET.tostring(server_uncustomized_elm)).object
    node_id = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'serverId':
            node_id = info.get('value')
    new_node = self.ex_get_node_by_id(node_id)
    return new_node