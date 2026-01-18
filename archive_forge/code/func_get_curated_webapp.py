from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_curated_webapp(self, resource_group, name, webapp):
    pip = self.serialize_obj(webapp, AZURE_OBJECT_CLASS)
    try:
        site_config = self.list_webapp_configuration(resource_group, name)
        app_settings = self.list_webapp_appsettings(resource_group, name)
        publish_cred = self.get_publish_credentials(resource_group, name)
        ftp_publish_url = self.get_webapp_ftp_publish_url(resource_group, name)
    except Exception:
        pass
    return self.construct_curated_webapp(webapp=pip, configuration=site_config, app_settings=app_settings, deployment_slot=None, ftp_publish_url=ftp_publish_url, publish_credentials=publish_cred)