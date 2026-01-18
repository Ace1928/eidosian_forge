import os
import sys
from breezy import branch, osutils, registry, tests
def test_lazy_import_registry(self):
    plugin_name = self.create_simple_plugin()
    a_registry = registry.Registry()
    a_registry.register_lazy('obj', plugin_name, 'object1')
    a_registry.register_lazy('function', plugin_name, 'function')
    a_registry.register_lazy('klass', plugin_name, 'MyClass')
    a_registry.register_lazy('module', plugin_name, None)
    self.assertEqual(['function', 'klass', 'module', 'obj'], sorted(a_registry.keys()))
    self.assertFalse(plugin_name in sys.modules)
    self.assertRaises(ImportError, a_registry.get, 'obj')
    plugin_path = self.test_dir + '/tmp'
    sys.path.append(plugin_path)
    try:
        obj = a_registry.get('obj')
        self.assertEqual('foo', obj)
        self.assertTrue(plugin_name in sys.modules)
        func = a_registry.get('function')
        self.assertEqual(plugin_name, func.__module__)
        self.assertEqual('function', func.__name__)
        self.assertEqual((1, [], '3'), func(1, [], '3'))
        klass = a_registry.get('klass')
        self.assertEqual(plugin_name, klass.__module__)
        self.assertEqual('MyClass', klass.__name__)
        inst = klass(1)
        self.assertIsInstance(inst, klass)
        self.assertEqual(1, inst.a)
        module = a_registry.get('module')
        self.assertIs(obj, module.object1)
        self.assertIs(func, module.function)
        self.assertIs(klass, module.MyClass)
    finally:
        sys.path.remove(plugin_path)