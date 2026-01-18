import urllib.parse
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def delete_all_tags(self):
    return self.manager.update_tags(self, [])