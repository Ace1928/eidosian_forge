from __future__ import absolute_import, division, print_function
import abc
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.module_utils.common._collections_compat import Sequence
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible.template import Templar
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.converter import (
@abc.abstractmethod
def setup_api(self):
    """
        This function needs to set up self.provider_information and self.api.
        It can indicate errors by raising DNSAPIError.
        """