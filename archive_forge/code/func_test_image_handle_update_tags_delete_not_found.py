from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_handle_update_tags_delete_not_found(self):
    self.my_image.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    props = self.stack.t.t['resources']['my_image']['properties'].copy()
    props['tags'] = ['tag1']
    self.my_image.t = self.my_image.t.freeze(properties=props)
    self.my_image.reparse()
    prop_diff = {'tags': ['tag2']}
    self.image_tags.delete.side_effect = exc.HTTPNotFound()
    self._handle_update_tags(prop_diff)