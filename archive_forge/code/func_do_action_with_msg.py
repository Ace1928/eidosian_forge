import uuid
import base64
from openstackclient.identity import common as identity_common
import os
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
import simplejson as json
import sys
from troveclient.apiclient import exceptions
def do_action_with_msg(action, success_msg):
    """Helper to run an action with return message."""
    action
    print(success_msg)