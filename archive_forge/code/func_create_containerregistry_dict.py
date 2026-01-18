from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_containerregistry_dict(registry, credentials):
    """
    Helper method to deserialize a ContainerRegistry to a dict
    :param: registry: return container registry object from Azure rest API call
    :param: credentials: return credential objects from Azure rest API call
    :return: dict of return container registry and it's credentials
    """
    results = dict(id=registry.id if registry is not None else '', name=registry.name if registry is not None else '', location=registry.location if registry is not None else '', admin_user_enabled=registry.admin_user_enabled if registry is not None else '', sku=registry.sku.name if registry is not None else '', provisioning_state=registry.provisioning_state if registry is not None else '', login_server=registry.login_server if registry is not None else '', credentials=dict(), tags=registry.tags if registry is not None else '')
    if credentials:
        results['credentials'] = dict(password=credentials.passwords[0].value, password2=credentials.passwords[1].value)
    return results