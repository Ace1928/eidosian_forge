import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def create_job_templates(app, client, mains_ids, libs_ids, parsed_args):
    args_dict = dict(name=parsed_args.name, type=parsed_args.type, mains=mains_ids, libs=libs_ids, description=parsed_args.description, interface=parsed_args.interface, is_public=parsed_args.public, is_protected=parsed_args.protected)
    if is_api_v2(app):
        data = client.job_templates.create(**args_dict).to_dict()
    else:
        data = client.jobs.create(**args_dict).to_dict()
    return data