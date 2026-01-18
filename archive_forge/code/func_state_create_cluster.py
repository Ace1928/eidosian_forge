from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def state_create_cluster(self):
    """
        Create cluster with given configuration
        """
    try:
        cluster_config_spec = vim.cluster.ConfigSpecEx()
        if not self.module.check_mode:
            self.datacenter.hostFolder.CreateClusterEx(self.cluster_name, cluster_config_spec)
        self.module.exit_json(changed=True)
    except vmodl.fault.InvalidArgument as invalid_args:
        self.module.fail_json(msg='Cluster configuration specification parameter is invalid : %s' % to_native(invalid_args.msg))
    except vim.fault.InvalidName as invalid_name:
        self.module.fail_json(msg="'%s' is an invalid name for a cluster : %s" % (self.cluster_name, to_native(invalid_name.msg)))
    except vmodl.fault.NotSupported as not_supported:
        self.module.fail_json(msg='Trying to create a cluster on an incorrect folder object : %s' % to_native(not_supported.msg))
    except vmodl.RuntimeFault as runtime_fault:
        self.module.fail_json(msg=to_native(runtime_fault.msg))
    except vmodl.MethodFault as method_fault:
        self.module.fail_json(msg=to_native(method_fault.msg))
    except Exception as generic_exc:
        self.module.fail_json(msg='Failed to create cluster due to generic exception %s' % to_native(generic_exc))