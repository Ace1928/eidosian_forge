from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_linux_profile_dict(linuxprofile):
    """
    Helper method to deserialize a ContainerServiceLinuxProfile to a dict
    :param: linuxprofile: ContainerServiceLinuxProfile with the Azure callback object
    :return: dict with the state on Azure
    """
    if linuxprofile:
        return dict(ssh_key=linuxprofile.ssh.public_keys[0].key_data, admin_username=linuxprofile.admin_username)
    else:
        return None