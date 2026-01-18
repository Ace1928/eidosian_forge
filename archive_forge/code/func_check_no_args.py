import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def check_no_args(command, args):
    options = collections.UserDict()
    no_args_error = False
    orig_img_service = glance_replicator.get_image_service
    try:
        glance_replicator.get_image_service = get_image_service
        command(options, args)
    except TypeError as e:
        if str(e) == 'Too few arguments.':
            no_args_error = True
    finally:
        glance_replicator.get_image_service = orig_img_service
    return no_args_error