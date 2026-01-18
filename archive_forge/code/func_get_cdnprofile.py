from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
def get_cdnprofile(self):
    """
        Gets the properties of the specified CDN profile.

        :return: deserialized CDN profile state dictionary
        """
    self.log('Checking if the CDN profile {0} is present'.format(self.name))
    try:
        response = self.cdn_client.profiles.get(self.resource_group, self.name)
        self.log('Response : {0}'.format(response))
        self.log('CDN profile : {0} found'.format(response.name))
        return cdnprofile_to_dict(response)
    except Exception:
        self.log('Did not find the CDN profile.')
        return False