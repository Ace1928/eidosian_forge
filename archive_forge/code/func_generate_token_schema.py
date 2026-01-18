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
def generate_token_schema(self, system_scoped=False, domain_scoped=False, project_scoped=False):
    """Return a dictionary of token properties to validate against."""
    ROLES_SCHEMA = {'type': 'array', 'items': {'type': 'object', 'properties': {'id': {'type': 'string'}, 'name': {'type': 'string'}, 'description': {'type': 'string'}, 'options': {'type': 'object'}}, 'required': ['id', 'name'], 'additionalProperties': False}, 'minItems': 1}
    properties = {'audit_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 2}, 'expires_at': {'type': 'string', 'pattern': unit.TIME_FORMAT_REGEX}, 'issued_at': {'type': 'string', 'pattern': unit.TIME_FORMAT_REGEX}, 'methods': {'type': 'array', 'items': {'type': 'string'}}, 'user': {'type': 'object', 'required': ['id', 'name', 'domain', 'password_expires_at'], 'properties': {'id': {'type': 'string'}, 'name': {'type': 'string'}, 'domain': {'type': 'object', 'properties': {'id': {'type': 'string'}, 'name': {'type': 'string'}}, 'required': ['id', 'name'], 'additonalProperties': False}, 'password_expires_at': {'type': ['string', 'null'], 'pattern': unit.TIME_FORMAT_REGEX}}, 'additionalProperties': False}}
    if system_scoped:
        properties['catalog'] = {'type': 'array'}
        properties['system'] = {'type': 'object', 'properties': {'all': {'type': 'boolean'}}}
        properties['roles'] = ROLES_SCHEMA
    elif domain_scoped:
        properties['catalog'] = {'type': 'array'}
        properties['roles'] = ROLES_SCHEMA
        properties['domain'] = {'type': 'object', 'required': ['id', 'name'], 'properties': {'id': {'type': 'string'}, 'name': {'type': 'string'}}, 'additionalProperties': False}
    elif project_scoped:
        properties['is_admin_project'] = {'type': 'boolean'}
        properties['catalog'] = {'type': 'array'}
        ROLES_SCHEMA['items']['properties']['domain_id'] = {'type': ['null', 'string']}
        properties['roles'] = ROLES_SCHEMA
        properties['is_domain'] = {'type': 'boolean'}
        properties['project'] = {'type': ['object'], 'required': ['id', 'name', 'domain'], 'properties': {'id': {'type': 'string'}, 'name': {'type': 'string'}, 'domain': {'type': ['object'], 'required': ['id', 'name'], 'properties': {'id': {'type': 'string'}, 'name': {'type': 'string'}}, 'additionalProperties': False}}, 'additionalProperties': False}
    schema = {'type': 'object', 'properties': properties, 'required': ['audit_ids', 'expires_at', 'issued_at', 'methods', 'user'], 'optional': [], 'additionalProperties': False}
    if system_scoped:
        schema['required'].extend(['system', 'roles'])
        schema['optional'].append('catalog')
    elif domain_scoped:
        schema['required'].extend(['domain', 'roles'])
        schema['optional'].append('catalog')
    elif project_scoped:
        schema['required'].append('project')
        schema['optional'].append('catalog')
        schema['optional'].append('OS-TRUST:trust')
        schema['optional'].append('is_admin_project')
        schema['optional'].append('is_domain')
    return schema