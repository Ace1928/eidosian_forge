from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def populate_project_id(self, context, res_dict, is_create):
    """Populate the owner information in a request body.

        Ensure both project_id and tenant_id attributes are present.
        Validate that the requestor has the required privileges.
        For a create request, copy owner info from context to request body
        if needed and verify that owner is specified if required.

        :param context: The request context.
        :param res_dict: The resource attributes from the request.
        :param attr_info: The attribute map for the resource.
        :param is_create: Is this a create request?
        :raises: HTTPBadRequest If neither the project_id nor tenant_id
            are specified in the res_dict.

        """
    populate_project_info(res_dict)
    _validate_privileges(context, res_dict)
    if is_create and 'project_id' not in res_dict:
        if context.project_id:
            res_dict['project_id'] = context.project_id
            res_dict['tenant_id'] = context.project_id
        elif 'tenant_id' in self.attributes:
            msg = _('Running without keystone AuthN requires that tenant_id is specified')
            raise exc.HTTPBadRequest(msg)