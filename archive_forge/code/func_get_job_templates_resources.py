import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def get_job_templates_resources(app, client, parsed_args):
    if is_api_v2(app):
        data = get_resource(client.job_templates, parsed_args.job_template).to_dict()
    else:
        data = get_resource(client.jobs, parsed_args.job_template).to_dict()
    return data