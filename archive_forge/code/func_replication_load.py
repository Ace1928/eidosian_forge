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
def replication_load(options, args):
    """%(prog)s load <server:port> <path>

    Load the contents of a local directory into glance.

    server:port: the location of the glance instance.
    path:        a directory on disk containing the data.
    """
    if len(args) < 2:
        raise TypeError(_('Too few arguments.'))
    path = args.pop()
    server, port = utils.parse_valid_host_port(args.pop())
    imageservice = get_image_service()
    client = imageservice(http.HTTPConnection(server, port), options.targettoken)
    updated = []
    for ent in os.listdir(path):
        if uuidutils.is_uuid_like(ent):
            image_uuid = ent
            LOG.info(_LI('Considering: %s'), image_uuid)
            meta_file_name = os.path.join(path, image_uuid)
            with open(meta_file_name) as meta_file:
                meta = jsonutils.loads(meta_file.read())
            for key in options.dontreplicate.split(' '):
                if key in meta:
                    LOG.debug('Stripping %(header)s from saved metadata', {'header': key})
                    del meta[key]
            if _image_present(client, image_uuid):
                LOG.debug('Image %s already present', image_uuid)
                headers = client.get_image_meta(image_uuid)
                for key in options.dontreplicate.split(' '):
                    if key in headers:
                        LOG.debug('Stripping %(header)s from target metadata', {'header': key})
                        del headers[key]
                if _dict_diff(meta, headers):
                    LOG.info(_LI('Image %s metadata has changed'), image_uuid)
                    headers, body = client.add_image_meta(meta)
                    _check_upload_response_headers(headers, body)
                    updated.append(meta['id'])
            else:
                if not os.path.exists(os.path.join(path, image_uuid + '.img')):
                    LOG.debug('%s dump is missing image data, skipping', image_uuid)
                    continue
                with open(os.path.join(path, image_uuid + '.img')) as img_file:
                    try:
                        headers, body = client.add_image(meta, img_file)
                        _check_upload_response_headers(headers, body)
                        updated.append(meta['id'])
                    except exc.HTTPConflict:
                        msg = _LE(IMAGE_ALREADY_PRESENT_MESSAGE) % image_uuid
                        LOG.error(msg)
    return updated