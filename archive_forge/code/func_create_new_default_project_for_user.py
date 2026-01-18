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
def create_new_default_project_for_user(self, user_id, domain_id, enable_project=True):
    ref = unit.new_project_ref(domain_id=domain_id, enabled=enable_project)
    r = self.post('/projects', body={'project': ref})
    project = self.assertValidProjectResponse(r, ref)
    body = {'user': {'default_project_id': project['id']}}
    r = self.patch('/users/%(user_id)s' % {'user_id': user_id}, body=body)
    self.assertValidUserResponse(r)
    return project