from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def fill_post_defaults(self, res_dict, exc_cls=lambda m: exceptions.InvalidInput(error_message=m), check_allow_post=True):
    """Fill in default values for attributes in a POST request.

        When a POST request is made, the attributes with default values do not
        need to be specified by the user. This function fills in the values of
        any unspecified attributes if they have a default value.

        If an attribute is not specified and it does not have a default value,
        an exception is raised.

        If an attribute is specified and it is not allowed in POST requests, an
        exception is raised. The caller can override this behavior by setting
        check_allow_post=False (used by some internal admin operations).

        :param res_dict: The resource attributes from the request.
        :param exc_cls: Exception to be raised on error that must take
            a single error message as it's only constructor arg.
        :param check_allow_post: Raises an exception if a non-POST-able
            attribute is specified.
        :raises: exc_cls If check_allow_post is True and this instance of
            ResourceAttributes doesn't support POST.
        """
    for attr, attr_vals in self.attributes.items():
        if attr_vals['allow_post']:
            value = _dict_populate_defaults(res_dict.get(attr, constants.ATTR_NOT_SPECIFIED), attr_vals)
            if value is not constants.ATTR_NOT_SPECIFIED:
                res_dict[attr] = value
            if 'default' not in attr_vals and attr not in res_dict:
                msg = _("Failed to parse request. Required attribute '%s' not specified") % attr
                raise exc_cls(msg)
            _fill_default(res_dict, attr, attr_vals)
        elif check_allow_post:
            if attr in res_dict:
                msg = _("Attribute '%s' not allowed in POST") % attr
                raise exc_cls(msg)