import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_invalid_hot_version(self):
    invalid_hot_version_tmp = template_format.parse('{\n            "heat_template_version" : "2012-12-12",\n            }')
    init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_hot_version_tmp)
    valid_versions = ['2013-05-23', '2014-10-16', '2015-04-30', '2015-10-15', '2016-04-08', '2016-10-14', '2017-02-24', '2017-09-01', '2018-03-02', '2018-08-31', '2021-04-16', 'newton', 'ocata', 'pike', 'queens', 'rocky', 'wallaby']
    ex_error_msg = 'The template version is invalid: "heat_template_version: 2012-12-12". "heat_template_version" should be one of: %s' % ', '.join(valid_versions)
    self.assertEqual(ex_error_msg, str(init_ex))