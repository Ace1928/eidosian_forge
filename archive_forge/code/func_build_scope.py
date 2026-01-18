from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils._text import to_native
def build_scope(self):
    subscription_scope = '/subscriptions/' + self.subscription_id
    if self.scope is None:
        return subscription_scope
    return self.scope