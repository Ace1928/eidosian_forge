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
def atom_from_string(base, value_string, symtab=None):
    type_ = base.type
    atom = None
    if type_ == ovs.db.types.IntegerType:
        atom = ovs.db.data.Atom(type_, int(value_string))
    elif type_ == ovs.db.types.RealType:
        atom = ovs.db.data.Atom(type_, ovs.db.parser.float_to_int(float(value_string)))
    elif type_ == ovs.db.types.BooleanType:
        if value_string in ('true', 'yes', 'on', '1'):
            atom = ovs.db.data.Atom(type_, True)
        elif value_string == ('false', 'no', 'off', '0'):
            atom = ovs.db.data.Atom(type_, False)
    elif type_ == ovs.db.types.StringType:
        atom = ovs.db.data.Atom(type_, value_string)
    elif type_ == ovs.db.types.UuidType:
        if value_string[0] == '@':
            assert symtab is not None
            uuid_ = symtab[value_string]
            atom = ovs.db.data.Atom(type_, uuid_)
        else:
            atom = ovs.db.data.Atom(type_, ovs.ovsuuid.from_string(value_string))
    if atom is None:
        raise ValueError('expected %s' % type_.to_string(), value_string)
    atom.check_constraints(base)
    return atom