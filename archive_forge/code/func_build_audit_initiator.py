import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def build_audit_initiator():
    """A pyCADF initiator describing the current authenticated context."""
    pycadf_host = host.Host(address=flask.request.remote_addr, agent=str(flask.request.user_agent))
    initiator = resource.Resource(typeURI=taxonomy.ACCOUNT_USER, host=pycadf_host)
    oslo_context = flask.request.environ.get(context.REQUEST_CONTEXT_ENV)
    if oslo_context.user_id:
        initiator.id = utils.resource_uuid(oslo_context.user_id)
        initiator.user_id = oslo_context.user_id
    if oslo_context.project_id:
        initiator.project_id = oslo_context.project_id
    if oslo_context.domain_id:
        initiator.domain_id = oslo_context.domain_id
    initiator.request_id = oslo_context.request_id
    if oslo_context.global_request_id:
        initiator.global_request_id = oslo_context.global_request_id
    return initiator