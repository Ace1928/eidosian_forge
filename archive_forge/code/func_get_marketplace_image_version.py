from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def get_marketplace_image_version(self):
    try:
        versions = self.compute_client.virtual_machine_images.list(self.location, self.image['publisher'], self.image['offer'], self.image['sku'])
    except ResourceNotFoundError as exc:
        self.fail('Error fetching image {0} {1} {2} - {3}'.format(self.image['publisher'], self.image['offer'], self.image['sku'], str(exc)))
    if versions and len(versions) > 0:
        if self.image['version'] == 'latest':
            return versions[len(versions) - 1]
        for version in versions:
            if version.name == self.image['version']:
                return version
    self.fail('Error could not find image {0} {1} {2} {3}'.format(self.image['publisher'], self.image['offer'], self.image['sku'], self.image['version']))