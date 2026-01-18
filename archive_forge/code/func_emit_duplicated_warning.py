import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
def emit_duplicated_warning(img):
    img_uuid_list = [str(image.id) for image in img]
    LOG.warning('Multiple matching images: %(img_uuid_list)s\nUsing image: %(chosen_one)s', {'img_uuid_list': img_uuid_list, 'chosen_one': img_uuid_list[0]})