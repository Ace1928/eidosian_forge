from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec,
from re import compile, match, sub
from time import sleep
def create_disk(self, disk, vmid, vm, vm_config):
    create = self.module.params['create']
    if create == 'disabled' and disk not in vm_config:
        return (False, 'Disk %s not found in VM %s and creation was disabled in parameters.' % (disk, vmid))
    if create == 'regular' and disk not in vm_config or create == 'forced':
        playbook_config = self.get_create_attributes()
        import_string = playbook_config.pop('import_from', None)
        iso_image = self.module.params.get('iso_image', None)
        if import_string:
            config_str = '%s:%s,import-from=%s' % (self.module.params['storage'], '0', import_string)
            timeout_str = 'Reached timeout while importing VM disk. Last line in task before timeout: %s'
            ok_str = 'Disk %s imported into VM %s'
        elif iso_image is not None:
            config_str = iso_image
            ok_str = 'CD-ROM was created on %s bus in VM %s'
        else:
            config_str = self.module.params['storage']
            if self.module.params.get('media') != 'cdrom':
                config_str += ':%s' % self.module.params['size']
            ok_str = 'Disk %s created in VM %s'
            timeout_str = 'Reached timeout while creating VM disk. Last line in task before timeout: %s'
        for k, v in playbook_config.items():
            config_str += ',%s=%s' % (k, v)
        disk_config_to_apply = {self.module.params['disk']: config_str}
    if create in ['disabled', 'regular'] and disk in vm_config:
        ok_str = 'Disk %s updated in VM %s'
        iso_image = self.module.params.get('iso_image', None)
        proxmox_config = disk_conf_str_to_dict(vm_config[disk])
        playbook_config = self.get_create_attributes()
        playbook_config.pop('import_from', None)
        if iso_image is not None:
            config_str = iso_image
        else:
            config_str = proxmox_config['volume']
        for k, v in playbook_config.items():
            config_str += ',%s=%s' % (k, v)
        for option in ['size', 'storage_name', 'volume', 'volume_name']:
            playbook_config.update({option: proxmox_config[option]})
        if iso_image is not None:
            playbook_config['volume'] = iso_image
        playbook_config = dict(((k, str(v)) for k, v in playbook_config.items()))
        if proxmox_config == playbook_config:
            return (False, 'Disk %s is up to date in VM %s' % (disk, vmid))
        disk_config_to_apply = {self.module.params['disk']: config_str}
    current_task_id = self.proxmox_api.nodes(vm['node']).qemu(vmid).config.post(**disk_config_to_apply)
    task_success = self.wait_till_complete_or_timeout(vm['node'], current_task_id)
    if task_success:
        return (True, ok_str % (disk, vmid))
    else:
        self.module.fail_json(msg=timeout_str % self.proxmox_api.nodes(vm['node']).tasks(current_task_id).log.get()[:1])