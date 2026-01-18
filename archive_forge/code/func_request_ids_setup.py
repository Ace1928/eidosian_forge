import abc
import contextlib
import copy
import hashlib
import os
import threading
from oslo_utils import reflection
from oslo_utils import strutils
import requests
from novaclient import exceptions
from novaclient import utils
def request_ids_setup(self):
    self.x_openstack_request_ids = []