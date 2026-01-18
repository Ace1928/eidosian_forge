import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_image(image_id=None, md5=NO_MD5, sha256=NO_SHA256, status='active', image_name=u'fake_image', data=None, checksum=u'ee36e35a297980dee1b514de9803ec6d'):
    if data:
        md5 = utils.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        with open(data, 'rb') as file_obj:
            for chunk in iter(lambda: file_obj.read(8192), b''):
                md5.update(chunk)
                sha256.update(chunk)
        md5 = md5.hexdigest()
        sha256 = sha256.hexdigest()
    return {u'image_state': u'available', u'container_format': u'bare', u'min_ram': 0, u'ramdisk_id': 'fake_ramdisk_id', u'updated_at': u'2016-02-10T05:05:02Z', u'file': '/v2/images/' + image_id + '/file', u'size': 3402170368, u'image_type': u'snapshot', u'disk_format': u'qcow2', u'id': image_id, u'schema': u'/v2/schemas/image', u'status': status, u'tags': [], u'visibility': u'private', u'locations': [{u'url': u'http://127.0.0.1/images/' + image_id, u'metadata': {}}], u'min_disk': 40, u'virtual_size': None, u'name': image_name, u'checksum': md5 or checksum, u'created_at': u'2016-02-10T05:03:11Z', u'owner_specified.openstack.md5': md5 or NO_MD5, u'owner_specified.openstack.sha256': sha256 or NO_SHA256, u'owner_specified.openstack.object': 'images/{name}'.format(name=image_name), u'protected': False}