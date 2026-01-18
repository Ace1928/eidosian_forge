from __future__ import absolute_import, division, print_function
def create_sas_policy(self):
    if self.rights == 'listen_send':
        rights = ['Listen', 'Send']
    elif self.rights == 'manage':
        rights = ['Listen', 'Send', 'Manage']
    else:
        rights = [str.capitalize(self.rights)]
    try:
        client = self._get_client()
        if self.queue or self.topic:
            rule = client.create_or_update_authorization_rule(self.resource_group, self.namespace, self.queue or self.topic, self.name, parameters={'rights': rights})
        else:
            rule = client.create_or_update_authorization_rule(self.resource_group, self.namespace, self.name, parameters={'rights': rights})
        return rule
    except Exception as exc:
        self.fail('Error when creating or updating SAS policy {0} - {1}'.format(self.name, exc.message or str(exc)))
    return None