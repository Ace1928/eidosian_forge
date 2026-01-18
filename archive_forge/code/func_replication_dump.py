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
def replication_dump(options, args):
    """%(prog)s dump <server:port> <path>

    Dump the contents of a glance instance to local disk.

    server:port: the location of the glance instance.
    path:        a directory on disk to contain the data.
    """
    if len(args) < 2:
        raise TypeError(_('Too few arguments.'))
    path = args.pop()
    server, port = utils.parse_valid_host_port(args.pop())
    imageservice = get_image_service()
    client = imageservice(http.HTTPConnection(server, port), options.sourcetoken)
    for image in client.get_images():
        LOG.debug('Considering: %(image_id)s (%(image_name)s) (%(image_size)d bytes)', {'image_id': image['id'], 'image_name': image.get('name', '--unnamed--'), 'image_size': image['size']})
        data_path = os.path.join(path, image['id'])
        data_filename = data_path + '.img'
        if not os.path.exists(data_path):
            LOG.info(_LI('Storing: %(image_id)s (%(image_name)s) (%(image_size)d bytes) in %(data_filename)s'), {'image_id': image['id'], 'image_name': image.get('name', '--unnamed--'), 'image_size': image['size'], 'data_filename': data_filename})
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(jsonutils.dumps(image))
            if image['status'] == 'active' and (not options.metaonly):
                LOG.debug('Image %s is active', image['id'])
                image_response = client.get_image(image['id'])
                with open(data_filename, 'wb') as f:
                    while True:
                        chunk = image_response.read(options.chunksize)
                        if not chunk:
                            break
                        f.write(chunk)