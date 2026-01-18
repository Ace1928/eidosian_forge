from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def find_datastore_cluster_by_name(self, datastore_cluster_name, datacenter=None, folder=None):
    """
        Get datastore cluster managed object by name
        Args:
            datastore_cluster_name: Name of datastore cluster
            datacenter: Managed object of the datacenter
            folder: Managed object of the folder which holds datastore

        Returns: Datastore cluster managed object if found else None

        """
    if datacenter and hasattr(datacenter, 'datastoreFolder'):
        folder = datacenter.datastoreFolder
    if not folder:
        folder = self.content.rootFolder
    data_store_clusters = get_all_objs(self.content, [vim.StoragePod], folder=folder)
    for dsc in data_store_clusters:
        if dsc.name == datastore_cluster_name:
            return dsc
    return None