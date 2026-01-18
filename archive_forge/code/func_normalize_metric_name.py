import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def normalize_metric_name(name):
    name = name.replace('.', '_')
    name = name.replace(':', '_')
    return name