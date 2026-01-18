from __future__ import absolute_import, print_function, unicode_literals
import re
import time
import unicodedata
from datetime import datetime
from .enums import ResourceType
from .permissions import Permissions
def decode_linux(line, match):
    ty, perms, links, uid, gid, size, mtime, name = match.groups()
    is_link = ty == 'l'
    is_dir = ty == 'd' or is_link
    if is_link:
        name, _, _link_name = name.partition('->')
        name = name.strip()
        _link_name = _link_name.strip()
    permissions = Permissions.parse(perms)
    mtime_epoch = _decode_linux_time(mtime)
    name = unicodedata.normalize('NFC', name)
    raw_info = {'basic': {'name': name, 'is_dir': is_dir}, 'details': {'size': int(size), 'type': int(ResourceType.directory if is_dir else ResourceType.file)}, 'access': {'permissions': permissions.dump()}, 'ftp': {'ls': line}}
    access = raw_info['access']
    details = raw_info['details']
    if mtime_epoch is not None:
        details['modified'] = mtime_epoch
    access['user'] = uid
    access['group'] = gid
    return raw_info