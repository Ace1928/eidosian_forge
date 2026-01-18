from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_package_info(self):
    hosts_info = {}
    for host in self.hosts:
        host_package_info = []
        host_pkg_mgr = host.configManager.imageConfigManager
        if host_pkg_mgr:
            pkgs = host_pkg_mgr.FetchSoftwarePackages()
            for pkg in pkgs:
                host_package_info.append(dict(name=pkg.name, version=pkg.version, vendor=pkg.vendor, summary=pkg.summary, description=pkg.description, acceptance_level=pkg.acceptanceLevel, maintenance_mode_required=pkg.maintenanceModeRequired, creation_date=pkg.creationDate))
        hosts_info[host.name] = host_package_info
    return hosts_info