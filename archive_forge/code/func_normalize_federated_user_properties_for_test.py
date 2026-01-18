import copy
import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
@staticmethod
def normalize_federated_user_properties_for_test(federated_user, email=None):
    federated_user['email'] = email
    federated_user['id'] = federated_user['unique_id']
    federated_user['name'] = federated_user['display_name']
    if not federated_user.get('domain'):
        federated_user['domain'] = {'id': uuid.uuid4().hex}