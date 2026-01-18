from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def gather_vm_facts(module, vm_params):
    """Gathers VM facts.

    Args:
        module: Reference to Ansible module object.
        vm_params (dict): A dictionary with VM parameters as returned
            by gather_vm_params() function.

    Returns:
        dict: VM facts.
    """
    if not vm_params:
        return {}
    xapi_session = XAPI.connect(module)
    vm_facts = {'state': xapi_to_module_vm_power_state(vm_params['power_state'].lower()), 'name': vm_params['name_label'], 'name_desc': vm_params['name_description'], 'uuid': vm_params['uuid'], 'is_template': vm_params['is_a_template'], 'folder': vm_params['other_config'].get('folder', ''), 'hardware': {'num_cpus': int(vm_params['VCPUs_max']), 'num_cpu_cores_per_socket': int(vm_params['platform'].get('cores-per-socket', '1')), 'memory_mb': int(int(vm_params['memory_dynamic_max']) / 1048576)}, 'disks': [], 'cdrom': {}, 'networks': [], 'home_server': vm_params['affinity'].get('name_label', ''), 'domid': vm_params['domid'], 'platform': vm_params['platform'], 'other_config': vm_params['other_config'], 'xenstore_data': vm_params['xenstore_data'], 'customization_agent': vm_params['customization_agent']}
    for vm_vbd_params in vm_params['VBDs']:
        if vm_vbd_params['type'] == 'Disk':
            vm_disk_sr_params = xapi_session.xenapi.SR.get_record(vm_vbd_params['VDI']['SR'])
            vm_disk_params = {'size': int(vm_vbd_params['VDI']['virtual_size']), 'name': vm_vbd_params['VDI']['name_label'], 'name_desc': vm_vbd_params['VDI']['name_description'], 'sr': vm_disk_sr_params['name_label'], 'sr_uuid': vm_disk_sr_params['uuid'], 'os_device': vm_vbd_params['device'], 'vbd_userdevice': vm_vbd_params['userdevice']}
            vm_facts['disks'].append(vm_disk_params)
        elif vm_vbd_params['type'] == 'CD':
            if vm_vbd_params['empty']:
                vm_facts['cdrom'].update(type='none')
            else:
                vm_facts['cdrom'].update(type='iso')
                vm_facts['cdrom'].update(iso_name=vm_vbd_params['VDI']['name_label'])
    for vm_vif_params in vm_params['VIFs']:
        vm_guest_metrics_networks = vm_params['guest_metrics'].get('networks', {})
        vm_network_params = {'name': vm_vif_params['network']['name_label'], 'mac': vm_vif_params['MAC'], 'vif_device': vm_vif_params['device'], 'mtu': vm_vif_params['MTU'], 'ip': vm_guest_metrics_networks.get('%s/ip' % vm_vif_params['device'], ''), 'prefix': '', 'netmask': '', 'gateway': '', 'ip6': [vm_guest_metrics_networks[ipv6] for ipv6 in sorted(vm_guest_metrics_networks.keys()) if ipv6.startswith('%s/ipv6/' % vm_vif_params['device'])], 'prefix6': '', 'gateway6': ''}
        if vm_params['customization_agent'] == 'native':
            if vm_vif_params['ipv4_addresses'] and vm_vif_params['ipv4_addresses'][0]:
                vm_network_params['prefix'] = vm_vif_params['ipv4_addresses'][0].split('/')[1]
                vm_network_params['netmask'] = ip_prefix_to_netmask(vm_network_params['prefix'])
            vm_network_params['gateway'] = vm_vif_params['ipv4_gateway']
            if vm_vif_params['ipv6_addresses'] and vm_vif_params['ipv6_addresses'][0]:
                vm_network_params['prefix6'] = vm_vif_params['ipv6_addresses'][0].split('/')[1]
            vm_network_params['gateway6'] = vm_vif_params['ipv6_gateway']
        elif vm_params['customization_agent'] == 'custom':
            vm_xenstore_data = vm_params['xenstore_data']
            for f in ['prefix', 'netmask', 'gateway', 'prefix6', 'gateway6']:
                vm_network_params[f] = vm_xenstore_data.get('vm-data/networks/%s/%s' % (vm_vif_params['device'], f), '')
        vm_facts['networks'].append(vm_network_params)
    return vm_facts