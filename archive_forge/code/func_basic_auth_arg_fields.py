from __future__ import absolute_import, division, print_function
import datetime
import uuid
def basic_auth_arg_fields():
    fields = {'host': {'required': True, 'type': 'str'}, 'username': {'required': True, 'type': 'str'}, 'password': {'required': True, 'type': 'str', 'no_log': True}}
    return fields