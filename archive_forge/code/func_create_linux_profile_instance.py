from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_linux_profile_instance(self, linuxprofile):
    """
        Helper method to serialize a dict to a ContainerServiceLinuxProfile
        :param: linuxprofile: dict with the parameters to setup the ContainerServiceLinuxProfile
        :return: ContainerServiceLinuxProfile
        """
    return self.managedcluster_models.ContainerServiceLinuxProfile(admin_username=linuxprofile['admin_username'], ssh=self.managedcluster_models.ContainerServiceSshConfiguration(public_keys=[self.managedcluster_models.ContainerServiceSshPublicKey(key_data=str(linuxprofile['ssh_key']))]))