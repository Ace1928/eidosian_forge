from __future__ import (absolute_import, division, print_function)
import atexit
import datetime
import itertools
import json
import os
import re
import ssl
import sys
import uuid
from time import time
from jinja2 import Environment
from ansible.module_utils.six import integer_types, PY3
from ansible.module_utils.six.moves import configparser
def facts_from_proplist(self, vm):
    """Get specific properties instead of serializing everything"""
    rdata = {}
    for prop in self.guest_props:
        self.debugl('getting %s property for %s' % (prop, vm.name))
        key = prop
        if self.lowerkeys:
            key = key.lower()
        if '.' not in prop:
            vm_property = getattr(vm, prop)
            if isinstance(vm_property, vim.CustomFieldsManager.Value.Array):
                temp_vm_property = []
                for vm_prop in vm_property:
                    temp_vm_property.append({'key': vm_prop.key, 'value': vm_prop.value})
                rdata[key] = temp_vm_property
            else:
                rdata[key] = vm_property
        else:
            parts = prop.split('.')
            total = len(parts) - 1
            val = None
            lastref = rdata
            for idx, x in enumerate(parts):
                if isinstance(val, dict):
                    if x in val:
                        val = val.get(x)
                    elif x.lower() in val:
                        val = val.get(x.lower())
                else:
                    if not val:
                        try:
                            val = getattr(vm, x)
                        except AttributeError as e:
                            self.debugl(e)
                    else:
                        try:
                            val = getattr(val, x)
                        except AttributeError as e:
                            self.debugl(e)
                    val = self._process_object_types(val)
                if self.lowerkeys:
                    x = x.lower()
                if idx != total:
                    if x not in lastref:
                        lastref[x] = {}
                    lastref = lastref[x]
                else:
                    lastref[x] = val
    if self.args.debug:
        self.debugl('For %s' % vm.name)
        for key in list(rdata.keys()):
            if isinstance(rdata[key], dict):
                for ikey in list(rdata[key].keys()):
                    self.debugl("Property '%s.%s' has value '%s'" % (key, ikey, rdata[key][ikey]))
            else:
                self.debugl("Property '%s' has value '%s'" % (key, rdata[key]))
    return rdata