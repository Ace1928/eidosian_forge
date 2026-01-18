from keystoneauth1 import session
from heat.common import context
def create_trust_context(self):
    return context.RequestContext(username=self.username, password=self.password, is_admin=False, trust_id='atrust', trustor_user_id=self.user_id)