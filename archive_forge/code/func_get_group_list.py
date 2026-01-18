import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_group_list(value):
    """ group-list = mailbox-list / CFWS / obs-group-list
        obs-group-list = 1*([CFWS] ",") [CFWS]

    """
    group_list = GroupList()
    if not value:
        group_list.defects.append(errors.InvalidHeaderDefect('end of header before group-list'))
        return (group_list, value)
    leader = None
    if value and value[0] in CFWS_LEADER:
        leader, value = get_cfws(value)
        if not value:
            group_list.defects.append(errors.InvalidHeaderDefect('end of header in group-list'))
            group_list.append(leader)
            return (group_list, value)
        if value[0] == ';':
            group_list.append(leader)
            return (group_list, value)
    token, value = get_mailbox_list(value)
    if len(token.all_mailboxes) == 0:
        if leader is not None:
            group_list.append(leader)
        group_list.extend(token)
        group_list.defects.append(errors.ObsoleteHeaderDefect('group-list with empty entries'))
        return (group_list, value)
    if leader is not None:
        token[:0] = [leader]
    group_list.append(token)
    return (group_list, value)