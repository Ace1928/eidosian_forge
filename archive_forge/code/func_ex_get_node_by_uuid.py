import os
import re
import time
import platform
import mimetypes
import subprocess
from os.path import join as pjoin
from collections import defaultdict
from libcloud.utils.py3 import ET, ensure_string
from libcloud.compute.base import Node, NodeState, NodeDriver
from libcloud.compute.types import Provider
from libcloud.utils.networking import is_public_subnet
def ex_get_node_by_uuid(self, uuid):
    """
        Retrieve Node object for a domain with a provided uuid.

        :param  uuid: Uuid of the domain.
        :type   uuid: ``str``
        """
    domain = self._get_domain_for_uuid(uuid=uuid)
    node = self._to_node(domain=domain)
    return node