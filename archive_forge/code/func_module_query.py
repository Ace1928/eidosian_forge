import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def module_query(self, instance):
    """Query an instance about installed modules."""
    return self._modules_get(instance, from_guest=True)