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
def get_vm_or_template(self, template_name=None):
    """
        Find the virtual machine or virtual machine template using name
        used for cloning purpose.
        Args:
            template_name: Name of virtual machine or virtual machine template

        Returns: virtual machine or virtual machine template object

        """
    template_obj = None
    if not template_name:
        return template_obj
    if '/' in template_name:
        vm_obj_path = os.path.dirname(template_name)
        vm_obj_name = os.path.basename(template_name)
        template_obj = find_vm_by_id(self.content, vm_obj_name, vm_id_type='inventory_path', folder=vm_obj_path)
        if template_obj:
            return template_obj
    else:
        template_obj = find_vm_by_id(self.content, vm_id=template_name, vm_id_type='uuid')
        if template_obj:
            return template_obj
        objects = self.get_managed_objects_properties(vim_type=vim.VirtualMachine, properties=['name'])
        templates = []
        for temp_vm_object in objects:
            if len(temp_vm_object.propSet) != 1:
                continue
            for temp_vm_object_property in temp_vm_object.propSet:
                if temp_vm_object_property.val == template_name:
                    templates.append(temp_vm_object.obj)
                    break
        if len(templates) > 1:
            self.module.fail_json(msg='Multiple virtual machines or templates with same name [%s] found.' % template_name)
        elif templates:
            template_obj = templates[0]
    return template_obj