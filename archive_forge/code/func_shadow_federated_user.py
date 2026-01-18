import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
def shadow_federated_user(self, idp_id, protocol_id, user, group_ids=None):
    """Map a federated user to a user.

        :param idp_id: identity provider id
        :param protocol_id: protocol id
        :param user: User dictionary
        :param group_ids: list of group ids to add the user to

        :returns: dictionary of the mapped User entity
        """
    user_dict = self._shadow_federated_user(idp_id, protocol_id, user)
    if group_ids:
        for group_id in group_ids:
            LOG.info('Adding user [%s] to group [%s].', user_dict, group_id)
            PROVIDERS.shadow_users_api.add_user_to_group_expires(user_dict['id'], group_id)
    return user_dict