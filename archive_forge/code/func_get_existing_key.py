from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from re import findall
def get_existing_key(self):
    for keys in self.paginate(self.url):
        if keys:
            for i in keys:
                existing_key_id = str(i['id'])
                if i['key'].split() == self.key.split()[:2]:
                    return existing_key_id
                elif i['title'] == self.name and self.force:
                    return existing_key_id
        else:
            return None