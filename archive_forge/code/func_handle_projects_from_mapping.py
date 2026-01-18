import functools
import uuid
import flask
from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from urllib import parse
from keystone.auth import plugins as auth_plugins
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def handle_projects_from_mapping(shadow_projects, idp_domain_id, existing_roles, user, assignment_api, resource_api):
    for shadow_project in shadow_projects:
        configure_project_domain(shadow_project, idp_domain_id, resource_api)
        try:
            project = resource_api.get_project_by_name(shadow_project['name'], shadow_project['domain']['id'])
        except exception.ProjectNotFound:
            LOG.info('Project %(project_name)s does not exist. It will be automatically provisioning for user %(user_id)s.', {'project_name': shadow_project['name'], 'user_id': user['id']})
            project_ref = {'id': uuid.uuid4().hex, 'name': shadow_project['name'], 'domain_id': shadow_project['domain']['id']}
            project = resource_api.create_project(project_ref['id'], project_ref)
        shadow_roles = shadow_project['roles']
        for shadow_role in shadow_roles:
            assignment_api.create_grant(existing_roles[shadow_role['name']]['id'], user_id=user['id'], project_id=project['id'])