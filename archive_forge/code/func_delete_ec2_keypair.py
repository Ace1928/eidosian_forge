from keystoneauth1 import session
from heat.common import context
def delete_ec2_keypair(self, credential_id=None, user_id=None, access=None):
    if user_id == self.user_id and access == self.creds.access:
        self.creds = None
    else:
        raise Exception('Incorrect user_id or access')