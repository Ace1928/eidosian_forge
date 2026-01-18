from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_folder_by_name(self, datacenter_name, folder_name):
    """
        Returns the identifier of a folder
        with the mentioned names.
        """
    datacenter = self.get_datacenter_by_name(datacenter_name)
    if not datacenter:
        return None
    filter_spec = Folder.FilterSpec(type=Folder.Type.VIRTUAL_MACHINE, names=set([folder_name]), datacenters=set([datacenter]))
    folder_summaries = self.api_client.vcenter.Folder.list(filter_spec)
    folder = folder_summaries[0].folder if len(folder_summaries) > 0 else None
    return folder