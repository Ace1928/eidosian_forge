from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.placement_groups import BoundPlacementGroup
def present_placement_group(self):
    self._get_placement_group()
    if self.hcloud_placement_group is None:
        self._create_placement_group()
    else:
        self._update_placement_group()