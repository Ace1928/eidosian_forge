from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def import_vm(module, connection):
    vms_service = connection.system_service().vms_service()
    if search_by_name(vms_service, module.params['name']) is not None:
        return False
    events_service = connection.system_service().events_service()
    last_event = events_service.list(max=1)[0]
    external_type = [tmp for tmp in ['kvm', 'xen', 'vmware'] if module.params[tmp] is not None][0]
    external_vm = module.params[external_type]
    imports_service = connection.system_service().external_vm_imports_service()
    imported_vm = imports_service.add(otypes.ExternalVmImport(vm=otypes.Vm(name=module.params['name']), name=external_vm.get('name'), username=external_vm.get('username', 'test'), password=external_vm.get('password', 'test'), provider=otypes.ExternalVmProviderType(external_type), url=external_vm.get('url'), cluster=otypes.Cluster(name=module.params['cluster']) if module.params['cluster'] else None, storage_domain=otypes.StorageDomain(name=external_vm.get('storage_domain')) if external_vm.get('storage_domain') else None, sparse=external_vm.get('sparse', True), host=otypes.Host(name=module.params['host']) if module.params['host'] else None))
    vms_service = connection.system_service().vms_service()
    wait(service=vms_service.vm_service(imported_vm.vm.id), condition=lambda vm: len(events_service.list(from_=int(last_event.id), search='type=1152 and vm.id=%s' % vm.id)) > 0 if vm is not None else False, fail_condition=lambda vm: vm is None, timeout=module.params['timeout'], poll_interval=module.params['poll_interval'])
    return True