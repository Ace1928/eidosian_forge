from __future__ import absolute_import, division, print_function
def get_auth_rule(self):
    rule = None
    try:
        client = self._get_client()
        if self.queue or self.topic:
            rule = client.get_authorization_rule(self.resource_group, self.namespace, self.queue or self.topic, self.name)
        else:
            rule = client.get_authorization_rule(self.resource_group, self.namespace, self.name)
    except Exception:
        pass
    return rule