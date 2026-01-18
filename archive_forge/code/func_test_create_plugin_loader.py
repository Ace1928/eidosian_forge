import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_create_plugin_loader(self):
    val_a = uuid.uuid4().hex
    val_b = uuid.uuid4().hex
    loader = TestSplitLoader()
    plugin_a = loader.load_from_options(a=val_a)
    plugin_b = loader.load_from_options(b=val_b)
    self.assertIsInstance(plugin_a, PluginA)
    self.assertIsInstance(plugin_b, PluginB)
    self.assertEqual(val_a, plugin_a.val)
    self.assertEqual(val_b, plugin_b.val)