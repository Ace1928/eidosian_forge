import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_deprecated_args(self):
    user_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    user_domain_id = uuid.uuid4().hex
    project_domain_id = uuid.uuid4().hex
    ctx = context.RequestContext(user_id=user_id, project_id=project_id, domain_id=domain_id, user_domain_id=user_domain_id, project_domain_id=project_domain_id)
    self.assertEqual(0, len(self.warnings))
    self.assertEqual(user_id, ctx.user_id)
    self.assertEqual(project_id, ctx.project_id)
    self.assertEqual(domain_id, ctx.domain_id)
    self.assertEqual(user_domain_id, ctx.user_domain_id)
    self.assertEqual(project_domain_id, ctx.project_domain_id)
    self.assertEqual(0, len(self.warnings))
    self.assertEqual(user_id, ctx.user)
    self.assertEqual(1, len(self.warnings))
    self.assertEqual(domain_id, ctx.domain)
    self.assertEqual(2, len(self.warnings))
    self.assertEqual(user_domain_id, ctx.user_domain)
    self.assertEqual(3, len(self.warnings))
    self.assertEqual(project_domain_id, ctx.project_domain)
    self.assertEqual(4, len(self.warnings))