import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_executions_list_with_pagination(self):
    ex1 = self.execution_create(params='{0} -d "a"'.format(self.direct_wf['Name']))
    time.sleep(1)
    ex2 = self.execution_create(params='{0} -d "b"'.format(self.direct_wf['Name']))
    all_wf_ids = [self.get_field_value(ex1, 'ID'), self.get_field_value(ex2, 'ID')]
    wf_execs = self.mistral_cli(True, 'execution-list')
    self.assertEqual(2, len(wf_execs))
    self.assertEqual(set(all_wf_ids), set([ex['ID'] for ex in wf_execs]))
    wf_execs = self.mistral_cli(True, 'execution-list', params='--oldest --limit 1')
    self.assertEqual(1, len(wf_execs))
    not_expected = wf_execs[0]['ID']
    expected = [ex for ex in all_wf_ids if ex != wf_execs[0]['ID']][0]
    wf_execs = self.mistral_cli(True, 'execution-list', params='--marker %s' % not_expected)
    self.assertNotIn(not_expected, [ex['ID'] for ex in wf_execs])
    self.assertIn(expected, [ex['ID'] for ex in wf_execs])
    wf_execs = self.mistral_cli(True, 'execution-list', params='--sort_keys Description')
    self.assertEqual(set(all_wf_ids), set([ex['ID'] for ex in wf_execs]))
    wf_ex1_index = -1
    wf_ex2_index = -1
    for idx, ex in enumerate(wf_execs):
        if ex['ID'] == all_wf_ids[0]:
            wf_ex1_index = idx
        elif ex['ID'] == all_wf_ids[1]:
            wf_ex2_index = idx
    self.assertLess(wf_ex1_index, wf_ex2_index)
    wf_execs = self.mistral_cli(True, 'execution-list', params='--sort_keys Description --sort_dirs=desc')
    self.assertEqual(set(all_wf_ids), set([ex['ID'] for ex in wf_execs]))
    wf_ex1_index = -1
    wf_ex2_index = -1
    for idx, ex in enumerate(wf_execs):
        if ex['ID'] == all_wf_ids[0]:
            wf_ex1_index = idx
        elif ex['ID'] == all_wf_ids[1]:
            wf_ex2_index = idx
    self.assertGreater(wf_ex1_index, wf_ex2_index)