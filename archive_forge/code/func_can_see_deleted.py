import copy
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_config import cfg
from oslo_context import context
from glance.api import policy
@property
def can_see_deleted(self):
    """Admins can see deleted by default"""
    return self.show_deleted or self.is_admin