import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from swiftclient import client as swiftclient_client
from swiftclient import exceptions as swiftclient_exceptions
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import swift
from heat.engine import node_data
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template as templatem
from heat.tests import common
from heat.tests import utils
def cont_index(obj_name, num_version_hist):
    objects = [{'bytes': 11, 'last_modified': '2014-07-03T19:42:03.281640', 'hash': '9214b4e4460fcdb9f3a369941400e71e', 'name': '02b' + obj_name + '/1404416326.51383', 'content_type': 'application/octet-stream'}] * num_version_hist
    objects.append({'bytes': 8, 'last_modified': '2014-07-03T19:42:03.849870', 'hash': '9ab7c0738852d7dd6a2dc0b261edc300', 'name': obj_name, 'content_type': 'application/x-www-form-urlencoded'})
    return (container_header, objects)