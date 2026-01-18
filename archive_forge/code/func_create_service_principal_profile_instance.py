from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_service_principal_profile_instance(self, spnprofile):
    """
        Helper method to serialize a dict to a ManagedClusterServicePrincipalProfile
        :param: spnprofile: dict with the parameters to setup the ManagedClusterServicePrincipalProfile
        :return: ManagedClusterServicePrincipalProfile
        """
    return self.managedcluster_models.ManagedClusterServicePrincipalProfile(client_id=spnprofile['client_id'], secret=spnprofile['client_secret'])