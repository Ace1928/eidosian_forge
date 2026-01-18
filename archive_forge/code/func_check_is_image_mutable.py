from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def check_is_image_mutable(context, image):
    """Replicate the DB-layer admin-or-owner check for the API.

    Much of the API code depends on hard-coded admin-or-owner
    enforcement in the DB or authorization layer, as the policy layer
    is largely a no-op by default. During blueprint policy-refactor,
    we are trying to remove as much of that as possible, but in
    certain places we need to do that (if secure_rbac is not
    enabled). This transitional helper provides a way to do that
    enforcement where necessary.

    :param context: A RequestContext
    :param image: An ImageProxy
    :raises: exception.Forbidden if the context is not the owner or an admin
    """
    if context.is_admin:
        return
    if image.owner is None or context.owner is None or image.owner != context.owner:
        raise exception.Forbidden(_('You do not own this image'))