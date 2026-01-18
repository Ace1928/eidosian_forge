from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def find_collection_resource_or_fail(self, collection_name, **params):
    """ Searches the collection resource by the collection name and the param passed.

        Returns:
            the resource as an object if it exists in manageiq, Fail otherwise.
        """
    resource = self.find_collection_resource_by(collection_name, **params)
    if resource:
        return resource
    else:
        msg = '{collection_name} where {params} does not exist in manageiq'.format(collection_name=collection_name, params=str(params))
        self.module.fail_json(msg=msg)