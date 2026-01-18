from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def group_to_dict(self, object):
    return dict(object_id=object.object_id, display_name=object.display_name, mail_nickname=object.mail_nickname, mail_enabled=object.mail_enabled, security_enabled=object.security_enabled, mail=object.mail)