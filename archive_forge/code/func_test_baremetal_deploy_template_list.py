from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_baremetal_deploy_template_list(self):
    steps = [{'interface': 'bios', 'step': 'apply_configuration', 'args': {'settings': [{'name': 'LogicalProc', 'value': 'Enabled'}]}, 'priority': 150}]
    deploy_template1 = self.create_deploy_template(name='CUSTOM_DEPLOY_TEMPLATE1', steps=steps)
    deploy_template2 = self.create_deploy_template(name='CUSTOM_DEPLOY_TEMPLATE2', steps=steps)
    deploy_templates = self.conn.baremetal.deploy_templates()
    ids = [template.id for template in deploy_templates]
    self.assertIn(deploy_template1.id, ids)
    self.assertIn(deploy_template2.id, ids)
    deploy_templates_with_details = self.conn.baremetal.deploy_templates(details=True)
    for dp in deploy_templates_with_details:
        self.assertIsNotNone(dp.id)
        self.assertIsNotNone(dp.name)
    deploy_tempalte_with_fields = self.conn.baremetal.deploy_templates(fields=['uuid'])
    for dp in deploy_tempalte_with_fields:
        self.assertIsNotNone(dp.id)
        self.assertIsNone(dp.name)