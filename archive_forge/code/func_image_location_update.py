import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
@log_call
@utils.no_4byte_params
def image_location_update(context, image_id, location):
    loc_id = location.get('id')
    if loc_id is None:
        msg = _('The location data has an invalid ID: %d') % loc_id
        raise exception.Invalid(msg)
    deleted = location['status'] in ('deleted', 'pending_delete')
    updated_time = timeutils.utcnow()
    delete_time = updated_time if deleted else None
    updated = False
    for loc in DATA['locations']:
        if loc['id'] == loc_id and loc['image_id'] == image_id:
            loc.update({'value': location['url'], 'meta_data': location['metadata'], 'status': location['status'], 'deleted': deleted, 'updated_at': updated_time, 'deleted_at': delete_time})
            updated = True
            break
    if not updated:
        msg = _('No location found with ID %(loc)s from image %(img)s') % dict(loc=loc_id, img=image_id)
        LOG.warning(msg)
        raise exception.NotFound(msg)