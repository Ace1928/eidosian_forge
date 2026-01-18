from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib  # noqa: F401, pylint: disable=unused-import
from ansible.module_utils.six.moves import configparser
from os.path import expanduser
from uuid import UUID
def get_network_domain(self, locator, location):
    """
        Retrieve a network domain by its name or Id.
        """
    if is_uuid(locator):
        network_domain = self.driver.ex_get_network_domain(locator)
    else:
        matching_network_domains = [network_domain for network_domain in self.driver.ex_list_network_domains(location=location) if network_domain.name == locator]
        if matching_network_domains:
            network_domain = matching_network_domains[0]
        else:
            network_domain = None
    if network_domain:
        return network_domain
    raise UnknownNetworkError("Network '%s' could not be found" % locator)