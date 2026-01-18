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
def replication_compare(options, args):
    """%(prog)s compare <fromserver:port> <toserver:port>

    Compare the contents of fromserver with those of toserver.

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
    differences = {}
    for image in source_client.get_images():
        if _image_present(target_client, image['id']):
            headers = target_client.get_image_meta(image['id'])
            for key in options.dontreplicate.split(' '):
                if key in image:
                    LOG.debug('Stripping %(header)s from source metadata', {'header': key})
                    del image[key]
                if key in headers:
                    LOG.debug('Stripping %(header)s from target metadata', {'header': key})
                    del headers[key]
            for key in image:
                if image[key] != headers.get(key):
                    LOG.warning(_LW('%(image_id)s: field %(key)s differs (source is %(source_value)s, destination is %(target_value)s)'), {'image_id': image['id'], 'key': key, 'source_value': image[key], 'target_value': headers.get(key, 'undefined')})
                    differences[image['id']] = 'diff'
                else:
                    LOG.debug('%(image_id)s is identical', {'image_id': image['id']})
        elif image['status'] == 'active':
            LOG.warning(_LW('Image %(image_id)s ("%(image_name)s") entirely missing from the destination'), {'image_id': image['id'], 'image_name': image.get('name', '--unnamed')})
            differences[image['id']] = 'missing'
    return differences