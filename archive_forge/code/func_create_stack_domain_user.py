from keystoneauth1 import session
from heat.common import context
def create_stack_domain_user(self, username, project_id, password=None):
    return self.user_id