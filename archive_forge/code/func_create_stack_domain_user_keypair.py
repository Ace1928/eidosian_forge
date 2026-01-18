from keystoneauth1 import session
from heat.common import context
def create_stack_domain_user_keypair(self, user_id, project_id):
    return self.creds