from keystoneauth1 import session
from heat.common import context
def create_ec2_keypair(self, user_id):
    if user_id == self.user_id:
        return self.creds