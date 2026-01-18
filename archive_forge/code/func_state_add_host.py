from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_add_host(self):
    """Add ESXi host to a cluster of folder in vCenter"""
    changed = True
    result = None
    if self.module.check_mode:
        result = "Host would be connected to vCenter '%s'" % self.vcenter
    else:
        host_connect_spec = self.get_host_connect_spec()
        as_connected = self.params.get('add_connected')
        esxi_license = None
        resource_pool = None
        task = None
        if self.folder_name:
            self.folder = self.search_folder(self.folder_name)
            try:
                task = self.folder.AddStandaloneHost(spec=host_connect_spec, compResSpec=resource_pool, addConnected=as_connected, license=esxi_license)
            except vim.fault.InvalidLogin as invalid_login:
                self.module.fail_json(msg='Cannot authenticate with the host : %s' % to_native(invalid_login))
            except vim.fault.HostConnectFault as connect_fault:
                self.module.fail_json(msg='An error occurred during connect : %s' % to_native(connect_fault))
            except vim.fault.DuplicateName as duplicate_name:
                self.module.fail_json(msg='The folder already contains a host with the same name : %s' % to_native(duplicate_name))
            except vmodl.fault.InvalidArgument as invalid_argument:
                self.module.fail_json(msg='An argument was specified incorrectly : %s' % to_native(invalid_argument))
            except vim.fault.AlreadyBeingManaged as already_managed:
                self.module.fail_json(msg='The host is already being managed by another vCenter server : %s' % to_native(already_managed))
            except vmodl.fault.NotEnoughLicenses as not_enough_licenses:
                self.module.fail_json(msg='There are not enough licenses to add this host : %s' % to_native(not_enough_licenses))
            except vim.fault.NoHost as no_host:
                self.module.fail_json(msg='Unable to contact the host : %s' % to_native(no_host))
            except vmodl.fault.NotSupported as not_supported:
                self.module.fail_json(msg='The folder is not a host folder : %s' % to_native(not_supported))
            except vim.fault.NotSupportedHost as host_not_supported:
                self.module.fail_json(msg='The host is running a software version that is not supported : %s' % to_native(host_not_supported))
            except vim.fault.AgentInstallFailed as agent_install:
                self.module.fail_json(msg='Error during vCenter agent installation : %s' % to_native(agent_install))
            except vim.fault.AlreadyConnected as already_connected:
                self.module.fail_json(msg='The host is already connected to the vCenter server : %s' % to_native(already_connected))
            except vim.fault.SSLVerifyFault as ssl_fault:
                self.module.fail_json(msg='The host certificate could not be authenticated : %s' % to_native(ssl_fault))
        elif self.cluster_name:
            self.host, self.cluster = self.search_cluster(self.datacenter_name, self.cluster_name, self.esxi_hostname)
            try:
                task = self.cluster.AddHost_Task(spec=host_connect_spec, asConnected=as_connected, resourcePool=resource_pool, license=esxi_license)
            except vim.fault.InvalidLogin as invalid_login:
                self.module.fail_json(msg='Cannot authenticate with the host : %s' % to_native(invalid_login))
            except vim.fault.HostConnectFault as connect_fault:
                self.module.fail_json(msg='An error occurred during connect : %s' % to_native(connect_fault))
            except vim.fault.DuplicateName as duplicate_name:
                self.module.fail_json(msg='The cluster already contains a host with the same name : %s' % to_native(duplicate_name))
            except vim.fault.AlreadyBeingManaged as already_managed:
                self.module.fail_json(msg='The host is already being managed by another vCenter server : %s' % to_native(already_managed))
            except vmodl.fault.NotEnoughLicenses as not_enough_licenses:
                self.module.fail_json(msg='There are not enough licenses to add this host : %s' % to_native(not_enough_licenses))
            except vim.fault.NoHost as no_host:
                self.module.fail_json(msg='Unable to contact the host : %s' % to_native(no_host))
            except vim.fault.NotSupportedHost as host_not_supported:
                self.module.fail_json(msg='The host is running a software version that is not supported; It may still be possible to add the host as a stand-alone host : %s' % to_native(host_not_supported))
            except vim.fault.TooManyHosts as too_many_hosts:
                self.module.fail_json(msg='No additional hosts can be added to the cluster : %s' % to_native(too_many_hosts))
            except vim.fault.AgentInstallFailed as agent_install:
                self.module.fail_json(msg='Error during vCenter agent installation : %s' % to_native(agent_install))
            except vim.fault.AlreadyConnected as already_connected:
                self.module.fail_json(msg='The host is already connected to the vCenter server : %s' % to_native(already_connected))
            except vim.fault.SSLVerifyFault as ssl_fault:
                self.module.fail_json(msg='The host certificate could not be authenticated : %s' % to_native(ssl_fault))
        try:
            changed, result = wait_for_task(task)
            result = "Host connected to vCenter '%s'" % self.vcenter
        except TaskError as task_error:
            self.module.fail_json(msg="Failed to add host to vCenter '%s' : %s" % (self.vcenter, to_native(task_error)))
    self.module.exit_json(changed=changed, result=result)