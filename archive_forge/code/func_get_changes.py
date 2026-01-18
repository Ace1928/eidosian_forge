from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils import six
from ansible_collections.community.general.plugins.module_utils.xenserver import (
def get_changes(self):
    """Finds VM parameters that differ from specified ones.

        This method builds a dictionary with hierarchy of VM parameters
        that differ from those specified in module parameters.

        Returns:
            list: VM parameters that differ from those specified in
            module parameters.
        """
    if not self.exists():
        self.module.fail_json(msg='Called get_changes on non existing VM!')
    need_poweredoff = False
    if self.module.params['is_template']:
        need_poweredoff = True
    try:
        if self.vm_params['is_a_template'] and (not self.vm_params['is_a_snapshot']):
            self.module.fail_json(msg='VM check: targeted VM is a template! Template reconfiguration is not supported.')
        if self.vm_params['is_a_snapshot']:
            self.module.fail_json(msg='VM check: targeted VM is a snapshot! Snapshot reconfiguration is not supported.')
        config_changes = []
        if self.module.params['name'] is not None and self.module.params['name'] != self.vm_params['name_label']:
            if self.module.params['name']:
                config_changes.append('name')
            else:
                self.module.fail_json(msg='VM check name: VM name cannot be an empty string!')
        if self.module.params['name_desc'] is not None and self.module.params['name_desc'] != self.vm_params['name_description']:
            config_changes.append('name_desc')
        vm_other_config = self.vm_params['other_config']
        vm_folder = vm_other_config.get('folder', '')
        if self.module.params['folder'] is not None and self.module.params['folder'] != vm_folder:
            config_changes.append('folder')
        if self.module.params['home_server'] is not None:
            if self.module.params['home_server'] and (not self.vm_params['affinity'] or self.module.params['home_server'] != self.vm_params['affinity']['name_label']):
                get_object_ref(self.module, self.module.params['home_server'], uuid=None, obj_type='home server', fail=True, msg_prefix='VM check home_server: ')
                config_changes.append('home_server')
            elif not self.module.params['home_server'] and self.vm_params['affinity']:
                config_changes.append('home_server')
        config_changes_hardware = []
        if self.module.params['hardware']:
            num_cpus = self.module.params['hardware'].get('num_cpus')
            if num_cpus is not None:
                try:
                    num_cpus = int(num_cpus)
                except ValueError as e:
                    self.module.fail_json(msg='VM check hardware.num_cpus: parameter should be an integer value!')
                if num_cpus < 1:
                    self.module.fail_json(msg='VM check hardware.num_cpus: parameter should be greater than zero!')
                if num_cpus != int(self.vm_params['VCPUs_at_startup']):
                    config_changes_hardware.append('num_cpus')
                    need_poweredoff = True
            num_cpu_cores_per_socket = self.module.params['hardware'].get('num_cpu_cores_per_socket')
            if num_cpu_cores_per_socket is not None:
                try:
                    num_cpu_cores_per_socket = int(num_cpu_cores_per_socket)
                except ValueError as e:
                    self.module.fail_json(msg='VM check hardware.num_cpu_cores_per_socket: parameter should be an integer value!')
                if num_cpu_cores_per_socket < 1:
                    self.module.fail_json(msg='VM check hardware.num_cpu_cores_per_socket: parameter should be greater than zero!')
                if num_cpus and num_cpus % num_cpu_cores_per_socket != 0:
                    self.module.fail_json(msg='VM check hardware.num_cpus: parameter should be a multiple of hardware.num_cpu_cores_per_socket!')
                vm_platform = self.vm_params['platform']
                vm_cores_per_socket = int(vm_platform.get('cores-per-socket', 1))
                if num_cpu_cores_per_socket != vm_cores_per_socket:
                    config_changes_hardware.append('num_cpu_cores_per_socket')
                    need_poweredoff = True
            memory_mb = self.module.params['hardware'].get('memory_mb')
            if memory_mb is not None:
                try:
                    memory_mb = int(memory_mb)
                except ValueError as e:
                    self.module.fail_json(msg='VM check hardware.memory_mb: parameter should be an integer value!')
                if memory_mb < 1:
                    self.module.fail_json(msg='VM check hardware.memory_mb: parameter should be greater than zero!')
                if memory_mb != int(max(int(self.vm_params['memory_dynamic_max']), int(self.vm_params['memory_static_max'])) / 1048576):
                    config_changes_hardware.append('memory_mb')
                    need_poweredoff = True
        if config_changes_hardware:
            config_changes.append({'hardware': config_changes_hardware})
        config_changes_disks = []
        config_new_disks = []
        vbd_userdevices_allowed = self.xapi_session.xenapi.VM.get_allowed_VBD_devices(self.vm_ref)
        if self.module.params['disks']:
            vm_disk_params_list = [disk_params for disk_params in self.vm_params['VBDs'] if disk_params['type'] == 'Disk']
            if len(self.module.params['disks']) < len(vm_disk_params_list):
                self.module.fail_json(msg='VM check disks: provided disks configuration has less disks than the target VM (%d < %d)!' % (len(self.module.params['disks']), len(vm_disk_params_list)))
            if not vm_disk_params_list:
                vm_disk_userdevice_highest = '-1'
            else:
                vm_disk_userdevice_highest = vm_disk_params_list[-1]['userdevice']
            for position in range(len(self.module.params['disks'])):
                if position < len(vm_disk_params_list):
                    vm_disk_params = vm_disk_params_list[position]
                else:
                    vm_disk_params = None
                disk_params = self.module.params['disks'][position]
                disk_size = self.get_normalized_disk_size(self.module.params['disks'][position], 'VM check disks[%s]: ' % position)
                disk_name = disk_params.get('name')
                if disk_name is not None and (not disk_name):
                    self.module.fail_json(msg='VM check disks[%s]: disk name cannot be an empty string!' % position)
                if vm_disk_params and vm_disk_params['VDI']:
                    disk_changes = []
                    if disk_name and disk_name != vm_disk_params['VDI']['name_label']:
                        disk_changes.append('name')
                    disk_name_desc = disk_params.get('name_desc')
                    if disk_name_desc is not None and disk_name_desc != vm_disk_params['VDI']['name_description']:
                        disk_changes.append('name_desc')
                    if disk_size:
                        if disk_size > int(vm_disk_params['VDI']['virtual_size']):
                            disk_changes.append('size')
                            need_poweredoff = True
                        elif disk_size < int(vm_disk_params['VDI']['virtual_size']):
                            self.module.fail_json(msg='VM check disks[%s]: disk size is smaller than existing (%d bytes < %s bytes). Reducing disk size is not allowed!' % (position, disk_size, vm_disk_params['VDI']['virtual_size']))
                    config_changes_disks.append(disk_changes)
                else:
                    if not disk_size:
                        self.module.fail_json(msg='VM check disks[%s]: no valid disk size specification found!' % position)
                    disk_sr_uuid = disk_params.get('sr_uuid')
                    disk_sr = disk_params.get('sr')
                    if disk_sr_uuid is not None or disk_sr is not None:
                        get_object_ref(self.module, disk_sr, disk_sr_uuid, obj_type='SR', fail=True, msg_prefix='VM check disks[%s]: ' % position)
                    elif self.default_sr_ref == 'OpaqueRef:NULL':
                        self.module.fail_json(msg='VM check disks[%s]: no default SR found! You must specify SR explicitly.' % position)
                    if not vbd_userdevices_allowed:
                        self.module.fail_json(msg='VM check disks[%s]: maximum number of devices reached!' % position)
                    disk_userdevice = None
                    for userdevice in vbd_userdevices_allowed:
                        if int(userdevice) > int(vm_disk_userdevice_highest):
                            disk_userdevice = userdevice
                            vbd_userdevices_allowed.remove(userdevice)
                            vm_disk_userdevice_highest = userdevice
                            break
                    if disk_userdevice is None:
                        disk_userdevice = str(int(self.vm_params['VBDs'][-1]['userdevice']) + 1)
                        self.module.fail_json(msg='VM check disks[%s]: new disk position %s is out of bounds!' % (position, disk_userdevice))
                    config_new_disks.append((position, disk_userdevice))
        for disk_change in config_changes_disks:
            if disk_change:
                config_changes.append({'disks_changed': config_changes_disks})
                break
        if config_new_disks:
            config_changes.append({'disks_new': config_new_disks})
        config_changes_cdrom = []
        if self.module.params['cdrom']:
            vm_cdrom_params_list = [cdrom_params for cdrom_params in self.vm_params['VBDs'] if cdrom_params['type'] == 'CD']
            if not vm_cdrom_params_list and (not vbd_userdevices_allowed):
                self.module.fail_json(msg='VM check cdrom: maximum number of devices reached!')
            cdrom_type = self.module.params['cdrom'].get('type')
            cdrom_iso_name = self.module.params['cdrom'].get('iso_name')
            if not cdrom_type:
                if cdrom_iso_name:
                    cdrom_type = 'iso'
                elif cdrom_iso_name is not None:
                    cdrom_type = 'none'
                self.module.params['cdrom']['type'] = cdrom_type
            if cdrom_type and (not vm_cdrom_params_list or cdrom_type != self.get_cdrom_type(vm_cdrom_params_list[0])):
                config_changes_cdrom.append('type')
            if cdrom_type == 'iso':
                get_object_ref(self.module, cdrom_iso_name, uuid=None, obj_type='ISO image', fail=True, msg_prefix='VM check cdrom.iso_name: ')
                if cdrom_iso_name and (not vm_cdrom_params_list or not vm_cdrom_params_list[0]['VDI'] or cdrom_iso_name != vm_cdrom_params_list[0]['VDI']['name_label']):
                    config_changes_cdrom.append('iso_name')
        if config_changes_cdrom:
            config_changes.append({'cdrom': config_changes_cdrom})
        config_changes_networks = []
        config_new_networks = []
        vif_devices_allowed = self.xapi_session.xenapi.VM.get_allowed_VIF_devices(self.vm_ref)
        if self.module.params['networks']:
            if len(self.module.params['networks']) < len(self.vm_params['VIFs']):
                self.module.fail_json(msg='VM check networks: provided networks configuration has less interfaces than the target VM (%d < %d)!' % (len(self.module.params['networks']), len(self.vm_params['VIFs'])))
            if not self.vm_params['VIFs']:
                vif_device_highest = '-1'
            else:
                vif_device_highest = self.vm_params['VIFs'][-1]['device']
            for position in range(len(self.module.params['networks'])):
                if position < len(self.vm_params['VIFs']):
                    vm_vif_params = self.vm_params['VIFs'][position]
                else:
                    vm_vif_params = None
                network_params = self.module.params['networks'][position]
                network_name = network_params.get('name')
                if network_name is not None and (not network_name):
                    self.module.fail_json(msg='VM check networks[%s]: network name cannot be an empty string!' % position)
                if network_name:
                    get_object_ref(self.module, network_name, uuid=None, obj_type='network', fail=True, msg_prefix='VM check networks[%s]: ' % position)
                network_mac = network_params.get('mac')
                if network_mac is not None:
                    network_mac = network_mac.lower()
                    if not is_mac(network_mac):
                        self.module.fail_json(msg="VM check networks[%s]: specified MAC address '%s' is not valid!" % (position, network_mac))
                network_type = network_params.get('type')
                network_ip = network_params.get('ip')
                network_netmask = network_params.get('netmask')
                network_prefix = None
                if not network_type and network_ip:
                    network_type = 'static'
                if self.vm_params['customization_agent'] == 'native' and network_type and (network_type == 'dhcp'):
                    network_type = 'none'
                if network_type and network_type == 'static':
                    if network_ip is not None:
                        network_ip_split = network_ip.split('/')
                        network_ip = network_ip_split[0]
                        if network_ip and (not is_valid_ip_addr(network_ip)):
                            self.module.fail_json(msg="VM check networks[%s]: specified IPv4 address '%s' is not valid!" % (position, network_ip))
                        if len(network_ip_split) > 1:
                            network_prefix = network_ip_split[1]
                            if not is_valid_ip_prefix(network_prefix):
                                self.module.fail_json(msg="VM check networks[%s]: specified IPv4 prefix '%s' is not valid!" % (position, network_prefix))
                    if network_netmask is not None:
                        if not is_valid_ip_netmask(network_netmask):
                            self.module.fail_json(msg="VM check networks[%s]: specified IPv4 netmask '%s' is not valid!" % (position, network_netmask))
                        network_prefix = ip_netmask_to_prefix(network_netmask, skip_check=True)
                    elif network_prefix is not None:
                        network_netmask = ip_prefix_to_netmask(network_prefix, skip_check=True)
                if network_type:
                    network_params['type'] = network_type
                if network_ip:
                    network_params['ip'] = network_ip
                if network_netmask:
                    network_params['netmask'] = network_netmask
                if network_prefix:
                    network_params['prefix'] = network_prefix
                network_gateway = network_params.get('gateway')
                if network_gateway and (not is_valid_ip_addr(network_gateway)):
                    self.module.fail_json(msg="VM check networks[%s]: specified IPv4 gateway '%s' is not valid!" % (position, network_gateway))
                network_type6 = network_params.get('type6')
                network_ip6 = network_params.get('ip6')
                network_prefix6 = None
                if not network_type6 and network_ip6:
                    network_type6 = 'static'
                if self.vm_params['customization_agent'] == 'native' and network_type6 and (network_type6 == 'dhcp'):
                    network_type6 = 'none'
                if network_type6 and network_type6 == 'static':
                    if network_ip6 is not None:
                        network_ip6_split = network_ip6.split('/')
                        network_ip6 = network_ip6_split[0]
                        if network_ip6 and (not is_valid_ip6_addr(network_ip6)):
                            self.module.fail_json(msg="VM check networks[%s]: specified IPv6 address '%s' is not valid!" % (position, network_ip6))
                        if len(network_ip6_split) > 1:
                            network_prefix6 = network_ip6_split[1]
                            if not is_valid_ip6_prefix(network_prefix6):
                                self.module.fail_json(msg="VM check networks[%s]: specified IPv6 prefix '%s' is not valid!" % (position, network_prefix6))
                if network_type6:
                    network_params['type6'] = network_type6
                if network_ip6:
                    network_params['ip6'] = network_ip6
                if network_prefix6:
                    network_params['prefix6'] = network_prefix6
                network_gateway6 = network_params.get('gateway6')
                if network_gateway6 and (not is_valid_ip6_addr(network_gateway6)):
                    self.module.fail_json(msg="VM check networks[%s]: specified IPv6 gateway '%s' is not valid!" % (position, network_gateway6))
                if vm_vif_params and vm_vif_params['network']:
                    network_changes = []
                    if network_name and network_name != vm_vif_params['network']['name_label']:
                        network_changes.append('name')
                    if network_mac and network_mac != vm_vif_params['MAC'].lower():
                        network_changes.append('mac')
                    if self.vm_params['customization_agent'] == 'native':
                        if network_type and network_type != vm_vif_params['ipv4_configuration_mode'].lower():
                            network_changes.append('type')
                        if network_type and network_type == 'static':
                            if network_ip and (not vm_vif_params['ipv4_addresses'] or not vm_vif_params['ipv4_addresses'][0] or network_ip != vm_vif_params['ipv4_addresses'][0].split('/')[0]):
                                network_changes.append('ip')
                            if network_prefix and (not vm_vif_params['ipv4_addresses'] or not vm_vif_params['ipv4_addresses'][0] or network_prefix != vm_vif_params['ipv4_addresses'][0].split('/')[1]):
                                network_changes.append('prefix')
                                network_changes.append('netmask')
                            if network_gateway is not None and network_gateway != vm_vif_params['ipv4_gateway']:
                                network_changes.append('gateway')
                        if network_type6 and network_type6 != vm_vif_params['ipv6_configuration_mode'].lower():
                            network_changes.append('type6')
                        if network_type6 and network_type6 == 'static':
                            if network_ip6 and (not vm_vif_params['ipv6_addresses'] or not vm_vif_params['ipv6_addresses'][0] or network_ip6 != vm_vif_params['ipv6_addresses'][0].split('/')[0]):
                                network_changes.append('ip6')
                            if network_prefix6 and (not vm_vif_params['ipv6_addresses'] or not vm_vif_params['ipv6_addresses'][0] or network_prefix6 != vm_vif_params['ipv6_addresses'][0].split('/')[1]):
                                network_changes.append('prefix6')
                            if network_gateway6 is not None and network_gateway6 != vm_vif_params['ipv6_gateway']:
                                network_changes.append('gateway6')
                    elif self.vm_params['customization_agent'] == 'custom':
                        vm_xenstore_data = self.vm_params['xenstore_data']
                        if network_type and network_type != vm_xenstore_data.get('vm-data/networks/%s/type' % vm_vif_params['device'], 'none'):
                            network_changes.append('type')
                            need_poweredoff = True
                        if network_type and network_type == 'static':
                            if network_ip and network_ip != vm_xenstore_data.get('vm-data/networks/%s/ip' % vm_vif_params['device'], ''):
                                network_changes.append('ip')
                                need_poweredoff = True
                            if network_prefix and network_prefix != vm_xenstore_data.get('vm-data/networks/%s/prefix' % vm_vif_params['device'], ''):
                                network_changes.append('prefix')
                                network_changes.append('netmask')
                                need_poweredoff = True
                            if network_gateway is not None and network_gateway != vm_xenstore_data.get('vm-data/networks/%s/gateway' % vm_vif_params['device'], ''):
                                network_changes.append('gateway')
                                need_poweredoff = True
                        if network_type6 and network_type6 != vm_xenstore_data.get('vm-data/networks/%s/type6' % vm_vif_params['device'], 'none'):
                            network_changes.append('type6')
                            need_poweredoff = True
                        if network_type6 and network_type6 == 'static':
                            if network_ip6 and network_ip6 != vm_xenstore_data.get('vm-data/networks/%s/ip6' % vm_vif_params['device'], ''):
                                network_changes.append('ip6')
                                need_poweredoff = True
                            if network_prefix6 and network_prefix6 != vm_xenstore_data.get('vm-data/networks/%s/prefix6' % vm_vif_params['device'], ''):
                                network_changes.append('prefix6')
                                need_poweredoff = True
                            if network_gateway6 is not None and network_gateway6 != vm_xenstore_data.get('vm-data/networks/%s/gateway6' % vm_vif_params['device'], ''):
                                network_changes.append('gateway6')
                                need_poweredoff = True
                    config_changes_networks.append(network_changes)
                else:
                    if not network_name:
                        self.module.fail_json(msg='VM check networks[%s]: network name is required for new network interface!' % position)
                    if network_type and network_type == 'static' and network_ip and (not network_netmask):
                        self.module.fail_json(msg='VM check networks[%s]: IPv4 netmask or prefix is required for new network interface!' % position)
                    if network_type6 and network_type6 == 'static' and network_ip6 and (not network_prefix6):
                        self.module.fail_json(msg='VM check networks[%s]: IPv6 prefix is required for new network interface!' % position)
                    if self.vm_params['customization_agent'] == 'custom':
                        for parameter in ['type', 'ip', 'prefix', 'gateway', 'type6', 'ip6', 'prefix6', 'gateway6']:
                            if network_params.get(parameter):
                                need_poweredoff = True
                                break
                    if not vif_devices_allowed:
                        self.module.fail_json(msg='VM check networks[%s]: maximum number of network interfaces reached!' % position)
                    vif_device = str(int(vif_device_highest) + 1)
                    if vif_device not in vif_devices_allowed:
                        self.module.fail_json(msg='VM check networks[%s]: new network interface position %s is out of bounds!' % (position, vif_device))
                    vif_devices_allowed.remove(vif_device)
                    vif_device_highest = vif_device
                    config_new_networks.append((position, vif_device))
        for network_change in config_changes_networks:
            if network_change:
                config_changes.append({'networks_changed': config_changes_networks})
                break
        if config_new_networks:
            config_changes.append({'networks_new': config_new_networks})
        config_changes_custom_params = []
        if self.module.params['custom_params']:
            for position in range(len(self.module.params['custom_params'])):
                custom_param = self.module.params['custom_params'][position]
                custom_param_key = custom_param['key']
                custom_param_value = custom_param['value']
                if custom_param_key not in self.vm_params:
                    self.module.fail_json(msg="VM check custom_params[%s]: unknown VM param '%s'!" % (position, custom_param_key))
                if custom_param_value != self.vm_params[custom_param_key]:
                    config_changes_custom_params.append(position)
        if config_changes_custom_params:
            config_changes.append({'custom_params': config_changes_custom_params})
        if need_poweredoff:
            config_changes.append('need_poweredoff')
        return config_changes
    except XenAPI.Failure as f:
        self.module.fail_json(msg='XAPI ERROR: %s' % f.details)