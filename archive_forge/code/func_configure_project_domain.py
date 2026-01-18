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
def configure_project_domain(shadow_project, idp_domain_id, resource_api):
    """Configure federated projects domain.

    We set the domain to be the default (idp_domain_id) if the project
    from the attribute mapping comes without a domain.
    """
    LOG.debug('Processing domain for project: %s', shadow_project)
    domain = shadow_project.get('domain', {'id': idp_domain_id})
    if 'id' not in domain:
        db_domain = resource_api.get_domain_by_name(domain['name'])
        domain = {'id': db_domain.get('id')}
    shadow_project['domain'] = domain
    LOG.debug('Project [%s] domain ID was resolved to [%s]', shadow_project['name'], shadow_project['domain']['id'])