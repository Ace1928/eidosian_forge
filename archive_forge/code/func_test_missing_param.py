import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_missing_param(self):
    templ_missing_parameter = '\nheat_template_version: 2015-04-30\nparameters:\n  one:\n    type: string\nresources:\n  str:\n    type: OS::Heat::RandomString\noutputs:\n  here-it-is:\n    value:\n      not-important\n'
    template = yaml.safe_load(self.template)
    del template['resources']['thisone']['properties']['two']
    try:
        self.stack_create(template=yaml.safe_dump(template), environment=self.env, files={'facade.yaml': self.templ_facade, 'concrete.yaml': templ_missing_parameter}, expected_status='CREATE_FAILED')
    except heat_exceptions.HTTPBadRequest as exc:
        exp = 'ERROR: Required property two for facade OS::Thingy missing in provider'
        self.assertEqual(exp, str(exc))