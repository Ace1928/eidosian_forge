import http.client as http
import os
import sys
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
from webob import exc
from glance.common import config
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def replication_livecopy(options, args):
    """%(prog)s livecopy <fromserver:port> <toserver:port>

    Load the contents of one glance instance into another.

    fromserver:port: the location of the source glance instance.
    toserver:port:   the location of the target glance instance.
    """
    if len(args) < 2:
        raise TypeError(_('Too few arguments.'))
    imageservice = get_image_service()
    target_server, target_port = utils.parse_valid_host_port(args.pop())
    target_conn = http.HTTPConnection(target_server, target_port)
    target_client = imageservice(target_conn, options.targettoken)
    source_server, source_port = utils.parse_valid_host_port(args.pop())
    source_conn = http.HTTPConnection(source_server, source_port)
    source_client = imageservice(source_conn, options.sourcetoken)
    updated = []
    for image in source_client.get_images():
        LOG.debug('Considering %(id)s', {'id': image['id']})
        for key in options.dontreplicate.split(' '):
            if key in image:
                LOG.debug('Stripping %(header)s from source metadata', {'header': key})
                del image[key]
        if _image_present(target_client, image['id']):
            headers = target_client.get_image_meta(image['id'])
            if headers['status'] == 'active':
                for key in options.dontreplicate.split(' '):
                    if key in image:
                        LOG.debug('Stripping %(header)s from source metadata', {'header': key})
                        del image[key]
                    if key in headers:
                        LOG.debug('Stripping %(header)s from target metadata', {'header': key})
                        del headers[key]
                if _dict_diff(image, headers):
                    LOG.info(_LI('Image %(image_id)s (%(image_name)s) metadata has changed'), {'image_id': image['id'], 'image_name': image.get('name', '--unnamed--')})
                    headers, body = target_client.add_image_meta(image)
                    _check_upload_response_headers(headers, body)
                    updated.append(image['id'])
        elif image['status'] == 'active':
            LOG.info(_LI('Image %(image_id)s (%(image_name)s) (%(image_size)d bytes) is being synced'), {'image_id': image['id'], 'image_name': image.get('name', '--unnamed--'), 'image_size': image['size']})
            if not options.metaonly:
                image_response = source_client.get_image(image['id'])
                try:
                    headers, body = target_client.add_image(image, image_response)
                    _check_upload_response_headers(headers, body)
                    updated.append(image['id'])
                except exc.HTTPConflict:
                    LOG.error(_LE(IMAGE_ALREADY_PRESENT_MESSAGE) % image['id'])
    return updated