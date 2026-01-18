from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.argspec import (
def create_record_transformation_argspec():
    return ArgumentSpec(argument_spec=dict(txt_transformation=dict(type='str', default='unquoted', choices=['api', 'quoted', 'unquoted']), txt_character_encoding=dict(type='str', choices=['decimal', 'octal'])))