import hashlib
from oslo_utils.secretutils import md5
from oslotest import base
import glance_store.driver as driver
def test_neg_too_few_kw_args(self):
    self.assertRaises(TypeError, self.fake_store.add, self.img_file, self.img_size, self.fake_context, self.fake_verifier, image_id=self.img_id)