from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_deploy_template_patch(self):
    name = 'CUSTOM_HYPERTHREADING_ON'
    steps = [{'interface': 'bios', 'step': 'apply_configuration', 'args': {'settings': [{'name': 'LogicalProc', 'value': 'Enabled'}]}, 'priority': 150}]
    deploy_template = self.create_deploy_template(name=name, steps=steps)
    deploy_template = self.conn.baremetal.patch_deploy_template(deploy_template, dict(path='/extra/answer', op='add', value=42))
    self.assertEqual({'answer': 42}, deploy_template.extra)
    self.assertEqual(name, deploy_template.name)
    deploy_template = self.conn.baremetal.get_deploy_template(deploy_template.id)
    self.assertEqual({'answer': 42}, deploy_template.extra)