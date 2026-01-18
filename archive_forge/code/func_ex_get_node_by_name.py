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
def ex_get_node_by_name(self, name):
    """
        Retrieve Node object for a domain with a provided name.

        :param  name: Name of the domain.
        :type   name: ``str``
        """
    domain = self._get_domain_for_name(name=name)
    node = self._to_node(domain=domain)
    return node