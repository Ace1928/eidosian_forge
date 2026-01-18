import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_create_wf_with_tags(self):
    init_wfs = self.workflow_create(self.wf_def)
    wf_name = init_wfs[1]['Name']
    self.assertTableStruct(init_wfs, ['Name', 'Created at', 'Updated at', 'Tags'])
    created_wf_info = self.get_item_info(get_from=init_wfs, get_by='Name', value=wf_name)
    self.assertEqual('tag', created_wf_info['Tags'])