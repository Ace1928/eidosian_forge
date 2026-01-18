import collections.abc
import json
import typing as ty
from urllib import parse
from urllib import request
from openstack import exceptions
from openstack.orchestration.util import environment_format
from openstack.orchestration.util import template_format
from openstack.orchestration.util import utils
def ignore_if(key, value):
    if key == 'base_url':
        return True
    if isinstance(value, dict):
        return True
    if '::' in value:
        return True
    if key in ['hooks', 'restricted_actions']:
        return True