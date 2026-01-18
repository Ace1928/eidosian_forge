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
def get_managed_objects_properties(self, vim_type, properties=None):
    """
        Look up a Managed Object Reference in vCenter / ESXi Environment
        :param vim_type: Type of vim object e.g, for datacenter - vim.Datacenter
        :param properties: List of properties related to vim object e.g. Name
        :return: local content object
        """
    root_folder = self.content.rootFolder
    if properties is None:
        properties = ['name']
    mor = self.content.viewManager.CreateContainerView(root_folder, [vim_type], True)
    traversal_spec = vmodl.query.PropertyCollector.TraversalSpec(name='traversal_spec', path='view', skip=False, type=vim.view.ContainerView)
    property_spec = vmodl.query.PropertyCollector.PropertySpec(type=vim_type, all=False, pathSet=properties)
    object_spec = vmodl.query.PropertyCollector.ObjectSpec(obj=mor, skip=True, selectSet=[traversal_spec])
    filter_spec = vmodl.query.PropertyCollector.FilterSpec(objectSet=[object_spec], propSet=[property_spec], reportMissingObjectsInResults=False)
    return self.content.propertyCollector.RetrieveContents([filter_spec])