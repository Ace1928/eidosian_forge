from glance.hacking import checks
from glance.tests import utils
def test_dict_constructor_with_list_copy(self):
    self.assertEqual(1, len(list(checks.dict_constructor_with_list_copy('    dict([(i, connect_info[i])'))))
    self.assertEqual(1, len(list(checks.dict_constructor_with_list_copy('    attrs = dict([(k, _from_json(v))'))))
    self.assertEqual(1, len(list(checks.dict_constructor_with_list_copy('        type_names = dict((value, key) for key, value in'))))
    self.assertEqual(1, len(list(checks.dict_constructor_with_list_copy('   dict((value, key) for key, value in'))))
    self.assertEqual(1, len(list(checks.dict_constructor_with_list_copy('foo(param=dict((k, v) for k, v in bar.items()))'))))
    self.assertEqual(1, len(list(checks.dict_constructor_with_list_copy(' dict([[i,i] for i in range(3)])'))))
    self.assertEqual(1, len(list(checks.dict_constructor_with_list_copy('  dd = dict([i,i] for i in range(3))'))))
    self.assertEqual(0, len(list(checks.dict_constructor_with_list_copy('        create_kwargs = dict(snapshot=snapshot,'))))
    self.assertEqual(0, len(list(checks.dict_constructor_with_list_copy('      self._render_dict(xml, data_el, data.__dict__)'))))