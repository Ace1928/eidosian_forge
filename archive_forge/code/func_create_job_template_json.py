import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def create_job_template_json(app, client, **template):
    if is_api_v2(app):
        data = client.job_templates.create(**template).to_dict()
    else:
        data = client.jobs.create(**template).to_dict()
    return data