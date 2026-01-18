from __future__ import absolute_import, division, print_function
def regenerate_sas_key(self, key_type):
    try:
        client = self._get_client()
        key = str.capitalize(key_type) + 'Key'
        if self.queue or self.topic:
            client.regenerate_keys(self.resource_group, self.namespace, self.queue or self.topic, self.name, key)
        else:
            client.regenerate_keys(self.resource_group, self.namespace, self.name, key)
    except Exception as exc:
        self.fail("Error when generating SAS policy {0}'s key - {1}".format(self.name, exc.message or str(exc)))
    return None