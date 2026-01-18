import re
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
import glance.async_
from glance.common import exception
from glance.i18n import _, _LE, _LW
def size_checked_iter(response, image_meta, expected_size, image_iter, notifier):
    image_id = image_meta['id']
    bytes_written = 0

    def notify_image_sent_hook(env):
        image_send_notification(bytes_written, expected_size, image_meta, response.request, notifier)
    if 'eventlet.posthooks' in response.request.environ:
        response.request.environ['eventlet.posthooks'].append((notify_image_sent_hook, (), {}))
    try:
        for chunk in image_iter:
            yield chunk
            bytes_written += len(chunk)
    except Exception as err:
        with excutils.save_and_reraise_exception():
            msg = _LE('An error occurred reading from backend storage for image %(image_id)s: %(err)s') % {'image_id': image_id, 'err': err}
            LOG.error(msg)
    if expected_size != bytes_written:
        msg = _LE('Backend storage for image %(image_id)s disconnected after writing only %(bytes_written)d bytes') % {'image_id': image_id, 'bytes_written': bytes_written}
        LOG.error(msg)
        raise exception.GlanceException(_('Corrupt image download for image %(image_id)s') % {'image_id': image_id})