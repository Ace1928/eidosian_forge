from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import find_obj, vmware_argument_spec, PyVmomi
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_config_option_for_guest(self):
    results = {}
    guest_id = []
    datacenter_name = self.params.get('datacenter')
    cluster_name = self.params.get('cluster_name')
    esxi_host_name = self.params.get('esxi_hostname')
    if self.params.get('guest_id'):
        guest_id = [self.params.get('guest_id')]
    if not self.params.get('get_hardware_versions') and (not self.params.get('get_guest_os_ids')) and (not self.params.get('get_config_options')):
        self.module.exit_json(msg="Please set at least one of these parameters 'get_hardware_versions', 'get_guest_os_ids', 'get_config_options' to True to get the desired info.")
    if self.params.get('get_config_options') and len(guest_id) == 0:
        self.module.fail_json(msg="Please set 'guest_id' when 'get_config_options' is set to True, to get the VM recommended config option for specific guest OS.")
    datacenter = find_obj(self.content, [vim.Datacenter], datacenter_name)
    if not datacenter:
        self.module.fail_json(msg='Unable to find datacenter "%s"' % datacenter_name)
    if cluster_name:
        cluster = find_obj(self.content, [vim.ComputeResource], cluster_name, folder=datacenter)
        if not cluster:
            self.module.fail_json(msg='Unable to find cluster "%s"' % cluster_name)
    elif esxi_host_name:
        host = find_obj(self.content, [vim.HostSystem], esxi_host_name, folder=datacenter)
        if not host:
            self.module.fail_json(msg='Unable to find host "%s"' % esxi_host_name)
        self.target_host = host
        cluster = host.parent
    env_browser = cluster.environmentBrowser
    if env_browser is None:
        self.module.fail_json(msg='The environmentBrowser of the ComputeResource is None, so can not get the desired config option info, please check your vSphere environment.')
    support_create_list, default_config = self.get_hardware_versions(env_browser=env_browser)
    if self.params.get('get_hardware_versions'):
        results.update({'supported_hardware_versions': support_create_list, 'default_hardware_version': default_config})
    if self.params.get('get_guest_os_ids') or self.params.get('get_config_options'):
        hardware_version = self.params.get('hardware_version', '')
        if hardware_version and len(support_create_list) != 0 and (hardware_version not in support_create_list):
            self.module.fail_json(msg="Specified hardware version '%s' is not in the supported create list: %s" % (hardware_version, support_create_list))
        vm_config_option_all = self.get_config_option_by_spec(env_browser=env_browser, key=hardware_version)
        supported_gos_list = self.get_guest_id_list(guest_os_desc=vm_config_option_all)
        if self.params.get('get_guest_os_ids'):
            results.update({vm_config_option_all.version: supported_gos_list})
        if self.params.get('get_config_options') and len(guest_id) != 0:
            if supported_gos_list and guest_id[0] not in supported_gos_list:
                self.module.fail_json(msg="Specified guest ID '%s' is not in the supported guest ID list: '%s'" % (guest_id[0], supported_gos_list))
            vm_config_option_guest = self.get_config_option_by_spec(env_browser=env_browser, guest_id=guest_id, key=hardware_version)
            guest_os_options = vm_config_option_guest.guestOSDescriptor
            guest_os_option_dict = self.get_config_option_recommended(guest_os_desc=guest_os_options, hwv_version=vm_config_option_guest.version)
            results.update({'recommended_config_options': guest_os_option_dict})
    self.module.exit_json(changed=False, failed=False, instance=results)