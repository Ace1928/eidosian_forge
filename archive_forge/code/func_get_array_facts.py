from __future__ import absolute_import, division, print_function
from re import match
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
def get_array_facts(self):
    """Extract particular facts from the storage array graph"""
    facts = dict(facts_from_proxy=not self.is_embedded(), ssid=self.ssid)
    controller_reference_label = self.get_controllers()
    array_facts = None
    try:
        rc, array_facts = self.request('storage-systems/%s/graph' % self.ssid)
    except Exception as error:
        self.module.fail_json(msg='Failed to obtain facts from storage array with id [%s]. Error [%s]' % (self.ssid, str(error)))
    facts['netapp_storage_array'] = dict(name=array_facts['sa']['saData']['storageArrayLabel'], chassis_serial=array_facts['sa']['saData']['chassisSerialNumber'], firmware=array_facts['sa']['saData']['fwVersion'], wwn=array_facts['sa']['saData']['saId']['worldWideName'], segment_sizes=array_facts['sa']['featureParameters']['supportedSegSizes'], cache_block_sizes=array_facts['sa']['featureParameters']['cacheBlockSizes'])
    facts['netapp_controllers'] = [dict(name=controller_reference_label[controller['controllerRef']], serial=controller['serialNumber'].strip(), status=controller['status']) for controller in array_facts['controller']]
    facts['netapp_host_groups'] = [dict(id=group['id'], name=group['name']) for group in array_facts['storagePoolBundle']['cluster']]
    facts['netapp_hosts'] = [dict(group_id=host['clusterRef'], hosts_reference=host['hostRef'], id=host['id'], name=host['name'], host_type_index=host['hostTypeIndex'], posts=host['hostSidePorts']) for host in array_facts['storagePoolBundle']['host']]
    facts['netapp_host_types'] = [dict(type=host_type['hostType'], index=host_type['index']) for host_type in array_facts['sa']['hostSpecificVals'] if 'hostType' in host_type.keys() and host_type['hostType']]
    facts['snapshot_images'] = [dict(id=snapshot['id'], status=snapshot['status'], pit_capacity=snapshot['pitCapacity'], creation_method=snapshot['creationMethod'], reposity_cap_utilization=snapshot['repositoryCapacityUtilization'], active_cow=snapshot['activeCOW'], rollback_source=snapshot['isRollbackSource']) for snapshot in array_facts['highLevelVolBundle']['pit']]
    facts['netapp_disks'] = [dict(id=disk['id'], available=disk['available'], media_type=disk['driveMediaType'], status=disk['status'], usable_bytes=disk['usableCapacity'], tray_ref=disk['physicalLocation']['trayRef'], product_id=disk['productID'], firmware_version=disk['firmwareVersion'], serial_number=disk['serialNumber'].lstrip()) for disk in array_facts['drive']]
    facts['netapp_management_interfaces'] = [dict(controller=controller_reference_label[controller['controllerRef']], name=iface['ethernet']['interfaceName'], alias=iface['ethernet']['alias'], channel=iface['ethernet']['channel'], mac_address=iface['ethernet']['macAddr'], remote_ssh_access=iface['ethernet']['rloginEnabled'], link_status=iface['ethernet']['linkStatus'], ipv4_enabled=iface['ethernet']['ipv4Enabled'], ipv4_address_config_method=iface['ethernet']['ipv4AddressConfigMethod'].lower().replace('config', ''), ipv4_address=iface['ethernet']['ipv4Address'], ipv4_subnet_mask=iface['ethernet']['ipv4SubnetMask'], ipv4_gateway=iface['ethernet']['ipv4GatewayAddress'], ipv6_enabled=iface['ethernet']['ipv6Enabled'], dns_config_method=iface['ethernet']['dnsProperties']['acquisitionProperties']['dnsAcquisitionType'], dns_servers=iface['ethernet']['dnsProperties']['acquisitionProperties']['dnsServers'] if iface['ethernet']['dnsProperties']['acquisitionProperties']['dnsServers'] else [], ntp_config_method=iface['ethernet']['ntpProperties']['acquisitionProperties']['ntpAcquisitionType'], ntp_servers=iface['ethernet']['ntpProperties']['acquisitionProperties']['ntpServers'] if iface['ethernet']['ntpProperties']['acquisitionProperties']['ntpServers'] else []) for controller in array_facts['controller'] for iface in controller['netInterfaces']]
    facts['netapp_hostside_interfaces'] = [dict(fc=[dict(controller=controller_reference_label[controller['controllerRef']], channel=iface['fibre']['channel'], link_status=iface['fibre']['linkStatus'], current_interface_speed=strip_interface_speed(iface['fibre']['currentInterfaceSpeed']), maximum_interface_speed=strip_interface_speed(iface['fibre']['maximumInterfaceSpeed'])) for controller in array_facts['controller'] for iface in controller['hostInterfaces'] if iface['interfaceType'] == 'fc'], ib=[dict(controller=controller_reference_label[controller['controllerRef']], channel=iface['ib']['channel'], link_status=iface['ib']['linkState'], mtu=iface['ib']['maximumTransmissionUnit'], current_interface_speed=strip_interface_speed(iface['ib']['currentSpeed']), maximum_interface_speed=strip_interface_speed(iface['ib']['supportedSpeed'])) for controller in array_facts['controller'] for iface in controller['hostInterfaces'] if iface['interfaceType'] == 'ib'], iscsi=[dict(controller=controller_reference_label[controller['controllerRef']], iqn=iface['iscsi']['iqn'], link_status=iface['iscsi']['interfaceData']['ethernetData']['linkStatus'], ipv4_enabled=iface['iscsi']['ipv4Enabled'], ipv4_address=iface['iscsi']['ipv4Data']['ipv4AddressData']['ipv4Address'], ipv4_subnet_mask=iface['iscsi']['ipv4Data']['ipv4AddressData']['ipv4SubnetMask'], ipv4_gateway=iface['iscsi']['ipv4Data']['ipv4AddressData']['ipv4GatewayAddress'], ipv6_enabled=iface['iscsi']['ipv6Enabled'], mtu=iface['iscsi']['interfaceData']['ethernetData']['maximumFramePayloadSize'], current_interface_speed=strip_interface_speed(iface['iscsi']['interfaceData']['ethernetData']['currentInterfaceSpeed']), supported_interface_speeds=strip_interface_speed(iface['iscsi']['interfaceData']['ethernetData']['supportedInterfaceSpeeds'])) for controller in array_facts['controller'] for iface in controller['hostInterfaces'] if iface['interfaceType'] == 'iscsi'], sas=[dict(controller=controller_reference_label[controller['controllerRef']], channel=iface['sas']['channel'], current_interface_speed=strip_interface_speed(iface['sas']['currentInterfaceSpeed']), maximum_interface_speed=strip_interface_speed(iface['sas']['maximumInterfaceSpeed']), link_status=iface['sas']['iocPort']['state']) for controller in array_facts['controller'] for iface in controller['hostInterfaces'] if iface['interfaceType'] == 'sas'])]
    facts['netapp_driveside_interfaces'] = [dict(controller=controller_reference_label[controller['controllerRef']], interface_type=interface['interfaceType'], interface_speed=strip_interface_speed(interface[interface['interfaceType']]['maximumInterfaceSpeed'] if interface['interfaceType'] == 'sata' or interface['interfaceType'] == 'sas' or interface['interfaceType'] == 'fibre' else interface[interface['interfaceType']]['currentSpeed'] if interface['interfaceType'] == 'ib' else interface[interface['interfaceType']]['interfaceData']['maximumInterfaceSpeed'] if interface['interfaceType'] == 'iscsi' else 'unknown')) for controller in array_facts['controller'] for interface in controller['driveInterfaces']]
    facts['netapp_storage_pools'] = [dict(id=storage_pool['id'], name=storage_pool['name'], available_capacity=storage_pool['freeSpace'], total_capacity=storage_pool['totalRaidedSpace'], used_capacity=storage_pool['usedSpace']) for storage_pool in array_facts['volumeGroup']]
    all_volumes = list(array_facts['volume'])
    facts['netapp_volumes'] = [dict(id=v['id'], name=v['name'], parent_storage_pool_id=v['volumeGroupRef'], capacity=v['capacity'], is_thin_provisioned=v['thinProvisioned'], workload=v['metadata']) for v in all_volumes]
    workload_tags = None
    try:
        rc, workload_tags = self.request('storage-systems/%s/workloads' % self.ssid)
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve workload tags. Array [%s].' % self.ssid)
    facts['netapp_workload_tags'] = [dict(id=workload_tag['id'], name=workload_tag['name'], attributes=workload_tag['workloadAttributes']) for workload_tag in workload_tags]
    facts['netapp_volumes_by_initiators'] = dict()
    for mapping in array_facts['storagePoolBundle']['lunMapping']:
        for host in facts['netapp_hosts']:
            if mapping['mapRef'] == host['hosts_reference'] or mapping['mapRef'] == host['group_id']:
                if host['name'] not in facts['netapp_volumes_by_initiators'].keys():
                    facts['netapp_volumes_by_initiators'].update({host['name']: []})
                for volume in all_volumes:
                    if mapping['id'] in [volume_mapping['id'] for volume_mapping in volume['listOfMappings']]:
                        workload_name = ''
                        metadata = dict()
                        for volume_tag in volume['metadata']:
                            if volume_tag['key'] == 'workloadId':
                                for workload_tag in facts['netapp_workload_tags']:
                                    if volume_tag['value'] == workload_tag['id']:
                                        workload_name = workload_tag['name']
                                        metadata = dict(((entry['key'], entry['value']) for entry in workload_tag['attributes'] if entry['key'] != 'profileId'))
                        facts['netapp_volumes_by_initiators'][host['name']].append(dict(name=volume['name'], id=volume['id'], wwn=volume['wwn'], workload_name=workload_name, meta_data=metadata))
    features = [feature for feature in array_facts['sa']['capabilities']]
    features.extend([feature['capability'] for feature in array_facts['sa']['premiumFeatures'] if feature['isEnabled']])
    features = list(set(features))
    features.sort()
    facts['netapp_enabled_features'] = features
    return facts