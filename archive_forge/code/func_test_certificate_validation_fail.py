from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_certificate_validation_fail(self):
    props = self.props.copy()
    props['type'] = 'certificate'
    snippet = self.res_template.freeze(properties=props)
    res = self._create_resource('test', snippet, self.stack)
    msg = 'Unexpected properties: algorithm, bit_length, mode. Only these properties are allowed for certificate type of order: ca_id, name, profile, request_data, request_type, source_container_ref, subject_dn.'
    self.assertRaisesRegex(exception.StackValidationFailed, msg, res.validate)