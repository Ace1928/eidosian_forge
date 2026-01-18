from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
def import_tag(self, repository, tag, resource_group, registry, source_image):
    source_tag = get_tag(source_image['repository'], source_image['name'])
    dest_repo_name = repository if repository else source_image['repository']
    dest_tag_name = tag if tag else source_image['name']
    dest_tag = get_tag(dest_repo_name, dest_tag_name)
    creds = None if not source_image['credentials'] else ImportSourceCredentials(username=source_image['credentials']['username'], password=source_image['credentials']['password'])
    params = ImportImageParameters(target_tags=[dest_tag], source=ImportSource(registry_uri=source_image['registry_uri'], source_image=source_tag, credentials=creds))
    try:
        if not resource_group:
            resource_group = self.get_registry_resource_group(registry)
        self.log(f'Importing {source_tag} as {dest_tag} to {registry} in {resource_group}')
        poller = self.containerregistry_client.registries.begin_import_image(resource_group_name=resource_group, registry_name=registry, parameters=params)
        self.get_poller_result(poller)
    except Exception as e:
        self.fail(f'Could not import {source_tag} as {dest_tag} to {registry} in {resource_group} - {str(e)}')