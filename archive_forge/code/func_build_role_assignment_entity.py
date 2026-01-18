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
def build_role_assignment_entity(self, link=None, prior_role_link=None, **attribs):
    """Build and return a role assignment entity with provided attributes.

        Provided attributes are expected to contain: domain_id or project_id,
        user_id or group_id, role_id and, optionally, inherited_to_projects.
        """
    entity = {'links': {'assignment': link or self.build_role_assignment_link(**attribs)}}
    if attribs.get('domain_id'):
        entity['scope'] = {'domain': {'id': attribs['domain_id']}}
    elif attribs.get('system'):
        entity['scope'] = {'system': {'all': True}}
    else:
        entity['scope'] = {'project': {'id': attribs['project_id']}}
    if attribs.get('user_id'):
        entity['user'] = {'id': attribs['user_id']}
        if attribs.get('group_id'):
            entity['links']['membership'] = '/groups/%s/users/%s' % (attribs['group_id'], attribs['user_id'])
    else:
        entity['group'] = {'id': attribs['group_id']}
    entity['role'] = {'id': attribs['role_id']}
    if attribs.get('inherited_to_projects'):
        entity['scope']['OS-INHERIT:inherited_to'] = 'projects'
    if prior_role_link:
        entity['links']['prior_role'] = prior_role_link
    return entity