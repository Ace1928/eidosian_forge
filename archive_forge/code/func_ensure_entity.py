from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
@_exception2fail_json(msg='Failed to ensure entity state: {0}')
def ensure_entity(self, resource, desired_entity, current_entity, params=None, state=None, foreman_spec=None):
    """
        Ensure that a given entity has a certain state

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param desired_entity: Desired properties of the entity
        :type desired_entity: dict
        :param current_entity: Current properties of the entity or None if nonexistent
        :type current_entity: Union[dict,None]
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional
        :param state: Desired state of the entity (optionally taken from the module)
        :type state: str, optional
        :param foreman_spec: Description of the entity structure (optionally taken from module)
        :type foreman_spec: dict, optional

        :return: The new current state of the entity
        :rtype: Union[dict,None]
        """
    if state is None:
        state = self.state
    if foreman_spec is None:
        foreman_spec = self.foreman_spec
    else:
        foreman_spec, _dummy = _foreman_spec_helper(foreman_spec)
    updated_entity = None
    self.record_before(resource, _flatten_entity(current_entity, foreman_spec))
    if state == 'present_with_defaults':
        if current_entity is None:
            updated_entity = self._create_entity(resource, desired_entity, params, foreman_spec)
    elif state == 'present':
        if current_entity is None:
            updated_entity = self._create_entity(resource, desired_entity, params, foreman_spec)
        else:
            updated_entity = self._update_entity(resource, desired_entity, current_entity, params, foreman_spec)
    elif state == 'copied':
        if current_entity is not None:
            updated_entity = self._copy_entity(resource, desired_entity, current_entity, params)
    elif state == 'reverted':
        if current_entity is not None:
            updated_entity = self._revert_entity(resource, current_entity, params)
    elif state == 'new_snapshot':
        updated_entity = self._create_entity(resource, desired_entity, params, foreman_spec)
    elif state == 'absent':
        if current_entity is not None:
            updated_entity = self._delete_entity(resource, current_entity, params)
    else:
        self.fail_json(msg='Not a valid state: {0}'.format(state))
    self.record_after(resource, _flatten_entity(updated_entity, foreman_spec))
    self.record_after_full(resource, updated_entity)
    return updated_entity