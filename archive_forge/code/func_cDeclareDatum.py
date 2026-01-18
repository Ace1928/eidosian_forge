import functools
import re
import uuid
import ovs.db.parser
import ovs.db.types
import ovs.json
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.socket_util
from ovs.db import error
def cDeclareDatum(self, name):
    n = len(self.values)
    if n == 0:
        return ['static struct ovsdb_datum %s = { .n = 0 };']
    s = []
    if self.type.key.type == ovs.db.types.StringType:
        s += ['static struct json %s_key_strings[%d] = {' % (name, n)]
        for key in sorted(self.values):
            s += ['    { .type = JSON_STRING, .string = "%s", .count = 2 },' % escapeCString(key.value)]
        s += ['};']
        s += ['static union ovsdb_atom %s_keys[%d] = {' % (name, n)]
        for i in range(n):
            s += ['    { .s = &%s_key_strings[%d] },' % (name, i)]
        s += ['};']
    else:
        s = ['static union ovsdb_atom %s_keys[%d] = {' % (name, n)]
        for key in sorted(self.values):
            s += ['    { %s },' % key.cInitAtom(key)]
        s += ['};']
    if self.type.value:
        if self.type.value.type == ovs.db.types.StringType:
            s += ['static struct json %s_val_strings[%d] = {' % (name, n)]
            for k, v in sorted(self.values):
                s += ['    { .type = JSON_STRING, .string = "%s", .count = 2 },' % escapeCString(v.value)]
            s += ['};']
            s += ['static union ovsdb_atom %s_values[%d] = {' % (name, n)]
            for i in range(n):
                s += ['    { .s = &%s_val_strings[%d] },' % (name, i)]
            s += ['};']
        else:
            s = ['static union ovsdb_atom %s_values[%d] = {' % (name, n)]
            for k, v in sorted(self.values.items()):
                s += ['    { %s },' % v.cInitAtom(v)]
            s += ['};']
    s += ['static struct ovsdb_datum %s = {' % name]
    s += ['    .n = %d,' % n]
    s += ['    .keys = %s_keys,' % name]
    if self.type.value:
        s += ['    .values = %s_values,' % name]
    s += ['};']
    return s