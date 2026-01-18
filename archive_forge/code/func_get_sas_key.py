from __future__ import absolute_import, division, print_function
def get_sas_key(self):
    try:
        client = self._get_client()
        if self.queue or self.topic:
            return client.list_keys(self.resource_group, self.namespace, self.queue or self.topic, self.name).as_dict()
        else:
            return client.list_keys(self.resource_group, self.namespace, self.name).as_dict()
    except Exception:
        pass
    return None