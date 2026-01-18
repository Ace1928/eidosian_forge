from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_datacenter_by_name(self, datacenter_name):
    """
        Returns the identifier of a datacenter
        Note: The method assumes only one datacenter with the mentioned name.
        """
    filter_spec = Datacenter.FilterSpec(names=set([datacenter_name]))
    datacenter_summaries = self.api_client.vcenter.Datacenter.list(filter_spec)
    datacenter = datacenter_summaries[0].datacenter if len(datacenter_summaries) > 0 else None
    return datacenter