from __future__ import absolute_import, division, print_function
import re
from random import randint
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, \
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def sanitize_disk_inputs(self):
    """
        Check correctness of disk input provided by user

        Returns: A list of dictionary containing disk information

        """
    disks_data = list()
    if not self.desired_disks:
        self.module.exit_json(changed=False, msg="No disks provided for virtual machine '%s' for management." % self.vm.name)
    for disk_index, disk in enumerate(self.desired_disks):
        current_disk = dict(disk_index=disk_index, state='present', destroy=True, filename=None, datastore_cluster=None, datastore=None, autoselect_datastore=True, disk_unit_number=0, controller_number=0, disk_mode='persistent', disk_type='thick', sharing=False, bus_sharing='noSharing', cluster_disk=False)
        if disk['type'] is not None:
            current_disk['disk_type'] = disk['type']
        if current_disk['disk_type'] == 'vpmemdisk':
            if self.vm.runtime.powerState != vim.VirtualMachinePowerState.poweredOff:
                self.module.fail_json(msg='Please make sure VM is in powered off state before doing vPMem disk reconfiguration.')
            disk['datastore'] = None
            disk['autoselect_datastore'] = None
            disk['filename'] = None
            disk['disk_mode'] = None
        if disk['state'] is not None:
            current_disk['state'] = disk['state']
        if disk['scsi_type'] is not None and disk['controller_type'] is None:
            current_disk['controller_type'] = disk['scsi_type']
        elif disk['scsi_type'] is None and disk['controller_type'] is None:
            current_disk['controller_type'] = 'paravirtual'
        elif disk['controller_type'] is not None and disk['scsi_type'] is None:
            current_disk['controller_type'] = disk['controller_type']
        else:
            self.module.fail_json(msg="Please specify either 'scsi_type' or 'controller_type' for disk index [%s]." % disk_index)
        if current_disk['controller_type'] == 'ide':
            if self.vm.runtime.powerState != vim.VirtualMachinePowerState.poweredOff:
                self.module.fail_json(msg='Please make sure VM is in powered off state before doing IDE disk reconfiguration.')
        if disk['scsi_controller'] is not None and disk['controller_number'] is None and (disk['controller_type'] is None):
            temp_disk_controller = disk['scsi_controller']
        elif disk['controller_number'] is not None and disk['scsi_controller'] is None and (disk['scsi_type'] is None):
            temp_disk_controller = disk['controller_number']
        else:
            self.module.fail_json(msg="Please specify 'scsi_controller' with 'scsi_type', or 'controller_number' with 'controller_type' under disk parameter for disk index [%s], which is required while creating or configuring disk." % disk_index)
        try:
            disk_controller = int(temp_disk_controller)
        except ValueError:
            self.module.fail_json(msg="Invalid controller bus number '%s' specified for disk index [%s]" % (temp_disk_controller, disk_index))
        if current_disk['controller_type'] == 'ide' and disk_controller not in [0, 1]:
            self.module.fail_json(msg="Invalid controller bus number '%s' specified for disk index [%s], valid value is 0 or 1" % (disk_controller, disk_index))
        current_disk['controller_number'] = disk_controller
        try:
            temp_disk_unit_number = int(disk['unit_number'])
        except ValueError:
            self.module.fail_json(msg="Invalid Disk unit number ID '%s' specified at index [%s]." % (disk['unit_number'], disk_index))
        if current_disk['controller_type'] in self.device_helper.scsi_device_type.keys():
            hw_version = int(self.vm.config.version.split('-')[1])
            if current_disk['controller_type'] == 'paravirtual' and hw_version >= 14:
                if temp_disk_unit_number not in range(0, 64):
                    self.module.fail_json(msg='Invalid Disk unit number ID specified for disk [%s] at index [%s], please specify value between 0 to 64 only (excluding 7).' % (temp_disk_unit_number, disk_index))
                if temp_disk_unit_number == 7:
                    self.module.fail_json(msg='Invalid Disk unit number ID specified for disk at index [%s], please specify value other than 7 as it is reserved for SCSI Controller.' % disk_index)
            else:
                if temp_disk_unit_number not in range(0, 16):
                    self.module.fail_json(msg='Invalid Disk unit number ID specified for disk [%s] at index [%s], please specify value between 0 to 15 only (excluding 7).' % (temp_disk_unit_number, disk_index))
                if temp_disk_unit_number == 7:
                    self.module.fail_json(msg='Invalid Disk unit number ID specified for disk at index [%s], please specify value other than 7 as it is reserved for SCSI Controller.' % disk_index)
        elif current_disk['controller_type'] == 'sata' and temp_disk_unit_number not in range(0, 30):
            self.module.fail_json(msg='Invalid Disk unit number ID specified for SATA disk [%s] at index [%s], please specify value between 0 to 29' % (temp_disk_unit_number, disk_index))
        elif current_disk['controller_type'] == 'nvme' and temp_disk_unit_number not in range(0, 15):
            self.module.fail_json(msg='Invalid Disk unit number ID specified for NVMe disk [%s] at index [%s], please specify value between 0 to 14' % (temp_disk_unit_number, disk_index))
        elif current_disk['controller_type'] == 'ide' and temp_disk_unit_number not in [0, 1]:
            self.module.fail_json(msg='Invalid Disk unit number ID specified for IDE disk [%s] at index [%s], please specify value 0 or 1' % (temp_disk_unit_number, disk_index))
        current_disk['disk_unit_number'] = temp_disk_unit_number
        if current_disk['state'] == 'absent':
            current_disk['destroy'] = disk.get('destroy', True)
        elif current_disk['state'] == 'present':
            if disk['datastore'] is not None:
                if disk['autoselect_datastore'] is not None:
                    self.module.fail_json(msg="Please specify either 'datastore' or 'autoselect_datastore' for disk index [%s]" % disk_index)
                datastore_name = disk['datastore']
                datastore_cluster = find_obj(self.content, [vim.StoragePod], datastore_name)
                datastore = find_obj(self.content, [vim.Datastore], datastore_name)
                if datastore is None and datastore_cluster is None:
                    self.module.fail_json(msg="Failed to find datastore or datastore cluster named '%s' in given configuration." % disk['datastore'])
                if datastore_cluster:
                    current_disk['datastore_cluster'] = datastore_cluster
                elif datastore:
                    ds_datacenter = get_parent_datacenter(datastore)
                    if ds_datacenter.name != self.module.params['datacenter']:
                        self.module.fail_json(msg="Get datastore '%s' in datacenter '%s', not the configured datacenter '%s'" % (datastore.name, ds_datacenter.name, self.module.params['datacenter']))
                    current_disk['datastore'] = datastore
                current_disk['autoselect_datastore'] = False
            elif disk['autoselect_datastore'] is not None:
                datastores = get_all_objs(self.content, [vim.Datastore])
                if not datastores:
                    self.module.fail_json(msg="Failed to gather information about available datastores in given datacenter '%s'." % self.module.params['datacenter'])
                datastore = None
                datastore_freespace = 0
                for ds in datastores:
                    if ds.summary.freeSpace > datastore_freespace:
                        datastore = ds
                        datastore_freespace = ds.summary.freeSpace
                current_disk['datastore'] = datastore
            elif current_disk['disk_type'] == 'vpmemdisk':
                current_disk['datastore'] = None
                current_disk['autoselect_datastore'] = False
            if disk['filename'] is not None:
                current_disk['filename'] = disk['filename']
            if [x for x in disk.keys() if (x.startswith('size_') or x == 'size') and disk[x] is not None]:
                disk_size_parse_failed = False
                if disk['size'] is not None:
                    size_regex = re.compile('(\\d+(?:\\.\\d+)?)([tgmkTGMK][bB])')
                    disk_size_m = size_regex.match(disk['size'])
                    if disk_size_m:
                        expected = disk_size_m.group(1)
                        unit = disk_size_m.group(2)
                    else:
                        disk_size_parse_failed = True
                    try:
                        if re.match('\\d+\\.\\d+', expected):
                            expected = float(expected)
                        else:
                            expected = int(expected)
                    except (TypeError, ValueError, NameError):
                        disk_size_parse_failed = True
                else:
                    param = [x for x in disk.keys() if x.startswith('size_') and disk[x] is not None][0]
                    unit = param.split('_')[-1]
                    disk_size = disk[param]
                    if isinstance(disk_size, (float, int)):
                        disk_size = str(disk_size)
                    try:
                        if re.match('\\d+\\.\\d+', disk_size):
                            expected = float(disk_size)
                        else:
                            expected = int(disk_size)
                    except (TypeError, ValueError, NameError):
                        disk_size_parse_failed = True
                if disk_size_parse_failed:
                    self.module.fail_json(msg='Failed to parse disk size for disk index [%s], please review value provided using documentation.' % disk_index)
                disk_units = dict(tb=3, gb=2, mb=1, kb=0)
                unit = unit.lower()
                if unit in disk_units:
                    current_disk['size'] = expected * 1024 ** disk_units[unit]
                else:
                    self.module.fail_json(msg="%s is not a supported unit for disk size for disk index [%s]. Supported units are ['%s']." % (unit, disk_index, "', '".join(disk_units.keys())))
            elif current_disk['filename'] is None and disk['type'] != 'rdm':
                self.module.fail_json(msg='No size, size_kb, size_mb, size_gb or size_tb attribute found into disk index [%s] configuration.' % disk_index)
            if disk['disk_mode'] is not None:
                current_disk['disk_mode'] = disk['disk_mode']
            if current_disk['disk_type'] != 'vpmemdisk':
                current_disk['sharing'] = self.get_sharing(disk, current_disk['disk_type'], disk_index)
                if disk['shares'] is not None:
                    current_disk['shares'] = disk['shares']
                if disk['iolimit'] is not None:
                    current_disk['iolimit'] = disk['iolimit']
            if disk['type'] == 'rdm':
                compatibility_mode = disk.get('compatibility_mode', 'physicalMode')
                if compatibility_mode not in ['physicalMode', 'virtualMode']:
                    self.module.fail_json(msg="Invalid 'compatibility_mode' specified for disk index [%s]. Please specify'compatibility_mode' value from ['physicalMode', 'virtualMode']." % disk_index)
                current_disk['compatibility_mode'] = compatibility_mode
                if 'rdm_path' not in disk and 'filename' not in disk:
                    self.module.fail_json(msg="rdm_path and/or 'filename' needs must be specified when using disk type 'rdm'for disk index [%s]" % disk_index)
                else:
                    current_disk['rdm_path'] = disk.get('rdm_path')
                if disk['filename'] and disk['rdm_path'] is None and (disk['cluster_disk'] is False):
                    self.module.fail_json(msg=" 'filename' requires setting 'cluster_disk' to True when using disk type 'rdm' without a'rdm_path' for disk index [%s]" % disk_index)
                else:
                    current_disk['cluster_disk'] = disk.get('cluster_disk')
            if disk['bus_sharing']:
                bus_sharing = disk.get('bus_sharing', 'noSharing')
                if bus_sharing not in ['noSharing', 'physicalSharing', 'virtualSharing']:
                    self.module.fail_json(msg="Invalid SCSI 'bus_sharing' specied for disk index [%s]. Please specify 'bus_sharing' value from ['noSharing', 'physicalSharing', 'virtualSharing']." % disk_index)
                current_disk['bus_sharing'] = bus_sharing
        disks_data.append(current_disk)
    return disks_data