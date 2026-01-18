import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def build_role_assignment_link(self, **attribs):
    """Build and return a role assignment link with provided attributes.

        Provided attributes are expected to contain: domain_id or project_id,
        user_id or group_id, role_id and, optionally, inherited_to_projects.
        """
    if attribs.get('domain_id'):
        link = '/domains/' + attribs['domain_id']
    elif attribs.get('system'):
        link = '/system'
    else:
        link = '/projects/' + attribs['project_id']
    if attribs.get('user_id'):
        link += '/users/' + attribs['user_id']
    else:
        link += '/groups/' + attribs['group_id']
    link += '/roles/' + attribs['role_id']
    if attribs.get('inherited_to_projects'):
        return '/OS-INHERIT%s/inherited_to_projects' % link
    return link