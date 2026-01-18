from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def populate_project_info(attributes):
    """Ensure that both project_id and tenant_id attributes are present.

    If either project_id or tenant_id is present in attributes then ensure
    that both are present.

    If neither are present then attributes is not updated.

    :param attributes: A dictionary of resource/API attributes
        or API request/response dict.
    :returns: attributes (updated with project_id if applicable).
    :raises: HTTPBadRequest if the attributes project_id and tenant_id
        don't match.
    """
    if 'tenant_id' in attributes and 'project_id' not in attributes:
        attributes['project_id'] = attributes['tenant_id']
    elif 'project_id' in attributes and 'tenant_id' not in attributes:
        attributes['tenant_id'] = attributes['project_id']
    if attributes.get('project_id') != attributes.get('tenant_id'):
        msg = _("'project_id' and 'tenant_id' do not match")
        raise exc.HTTPBadRequest(msg)
    return attributes