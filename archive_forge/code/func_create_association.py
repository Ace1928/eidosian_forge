import uuid
from keystone import exception
def create_association(self, **kwargs):
    association = {'policy_id': uuid.uuid4().hex, 'endpoint_id': None, 'service_id': None, 'region_id': None}
    association.update(kwargs)
    self.driver.create_policy_association(**association)
    return association