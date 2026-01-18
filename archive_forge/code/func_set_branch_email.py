from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def set_branch_email(self, b, email):
    b.get_config_stack().set('email', email)