import re
from tempest.lib import exceptions
from glanceclient.tests.functional import base
def test_member_list_v2(self):
    try:
        self.glance('--os-image-api-version 2 image-create --name temp')
    except Exception:
        pass
    out = self.glance('--os-image-api-version 2 image-list --visibility private')
    image_list = self.parser.listing(out)
    if len(image_list) > 0:
        param_image_id = '--image-id %s' % image_list[0]['ID']
        out = self.glance('--os-image-api-version 2 member-list', params=param_image_id)
        endpoints = self.parser.listing(out)
        self.assertTableStruct(endpoints, ['Image ID', 'Member ID', 'Status'])
    else:
        param_image_id = '--image-id fake_image_id'
        self.assertRaises(exceptions.CommandFailed, self.glance, '--os-image-api-version 2 member-list', params=param_image_id)