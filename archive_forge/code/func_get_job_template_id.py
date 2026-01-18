import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def get_job_template_id(app, client, parsed_args):
    if is_api_v2(app):
        jt_id = get_resource_id(client.job_templates, parsed_args.job_template)
    else:
        jt_id = get_resource_id(client.jobs, parsed_args.job_template)
    return jt_id