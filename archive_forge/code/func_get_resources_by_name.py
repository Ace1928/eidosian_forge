from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
def get_resources_by_name(self, name):
    """Fetches the project resources with the given name.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        resources -- resources dictionary representation (or {} if error)
        """
    err_msg, project = self.get_by_name(name)
    if err_msg:
        return (err_msg, {})
    return self.get_resources_by_id(project.get('id'))