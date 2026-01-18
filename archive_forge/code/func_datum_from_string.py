import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
def datum_from_string(type_, value_string, symtab=None):
    value_string = value_string.strip()
    if type_.is_map():
        if value_string.startswith('{'):
            LOG.debug('value_string %s', value_string)
            raise NotImplementedError()
        d = dict((v.split('=', 1) for v in value_string.split(',')))
        d = dict(((atom_from_string(type_.key, key, symtab), atom_from_string(type_.value, value, symtab)) for key, value in d.items()))
    elif type_.is_set():
        if value_string.startswith('['):
            LOG.debug('value_string %s', value_string)
            raise NotImplementedError()
        values = value_string.split(',')
        d = dict(((atom_from_string(type_.key, value, symtab), None) for value in values))
    else:
        atom = atom_from_string(type_.key, value_string, symtab)
        d = {atom: None}
    datum = ovs.db.data.Datum(type_, d)
    return datum.to_json()