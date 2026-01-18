import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
def ex_delete_permissions(self, container_name, object_name=None, entity=None):
    """
        Delete permissions for an ACL entity on a container or object.

        :param container_name: The container name.
        :type container_name: ``str``

        :param object_name: The object name. Optional. Not providing an object
            will delete a container permission.
        :type object_name: ``str``

        :param entity: The entity to whose permission will be deleted.
            Optional. If not provided, the role will be applied to the
            authenticated user, if using an OAuth2 authentication scheme.
        :type entity: ``str`` or ``None``
        """
    object_name = _clean_object_name(object_name)
    if not entity:
        user_id = self._get_user()
        if not user_id:
            raise ValueError('Must provide an entity. Driver is not using an authenticated user.')
        else:
            entity = 'user-%s' % user_id
    if object_name:
        url = '/storage/v1/b/{}/o/{}/acl/{}'.format(container_name, object_name, entity)
    else:
        url = '/storage/v1/b/{}/acl/{}'.format(container_name, entity)
    self.json_connection.request(url, method='DELETE')