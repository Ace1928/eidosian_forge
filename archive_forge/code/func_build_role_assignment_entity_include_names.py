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
def build_role_assignment_entity_include_names(self, domain_ref=None, role_ref=None, group_ref=None, user_ref=None, project_ref=None, inherited_assignment=None):
    """Build and return a role assignment entity with provided attributes.

        The expected attributes are: domain_ref or project_ref,
        user_ref or group_ref, role_ref and, optionally, inherited_to_projects.
        """
    entity = {'links': {}}
    attributes_for_links = {}
    if project_ref:
        dmn_name = PROVIDERS.resource_api.get_domain(project_ref['domain_id'])['name']
        entity['scope'] = {'project': {'id': project_ref['id'], 'name': project_ref['name'], 'domain': {'id': project_ref['domain_id'], 'name': dmn_name}}}
        attributes_for_links['project_id'] = project_ref['id']
    else:
        entity['scope'] = {'domain': {'id': domain_ref['id'], 'name': domain_ref['name']}}
        attributes_for_links['domain_id'] = domain_ref['id']
    if user_ref:
        dmn_name = PROVIDERS.resource_api.get_domain(user_ref['domain_id'])['name']
        entity['user'] = {'id': user_ref['id'], 'name': user_ref['name'], 'domain': {'id': user_ref['domain_id'], 'name': dmn_name}}
        attributes_for_links['user_id'] = user_ref['id']
    else:
        dmn_name = PROVIDERS.resource_api.get_domain(group_ref['domain_id'])['name']
        entity['group'] = {'id': group_ref['id'], 'name': group_ref['name'], 'domain': {'id': group_ref['domain_id'], 'name': dmn_name}}
        attributes_for_links['group_id'] = group_ref['id']
    if role_ref:
        entity['role'] = {'id': role_ref['id'], 'name': role_ref['name']}
        if role_ref['domain_id']:
            dmn_name = PROVIDERS.resource_api.get_domain(role_ref['domain_id'])['name']
            entity['role']['domain'] = {'id': role_ref['domain_id'], 'name': dmn_name}
        attributes_for_links['role_id'] = role_ref['id']
    if inherited_assignment:
        entity['scope']['OS-INHERIT:inherited_to'] = 'projects'
        attributes_for_links['inherited_to_projects'] = True
    entity['links']['assignment'] = self.build_role_assignment_link(**attributes_for_links)
    return entity