from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def check_admin_or_same_owner(context, properties):
    """Check that legacy behavior on create with owner is preserved.

    Legacy behavior requires a static check that owner is not
    inconsistent with the context, unless the caller is an
    admin. Enforce that here, if needed.

    :param context: A RequestContext
    :param properties: The properties being used to create the image, which may
                       contain an owner
    :raises: exception.Forbidden if the context is not an admin and owner is
             set to something other than the context's project
    """
    if context.is_admin:
        return
    if context.project_id != properties.get('owner', context.project_id):
        msg = _("You are not permitted to create images owned by '%s'.")
        raise exception.Forbidden(msg % properties['owner'])