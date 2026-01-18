import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
@unit.skip_if_cache_disabled('identity')
def test_invalidate_cache_when_purge_mappings(self):
    local_id1 = uuid.uuid4().hex
    local_id2 = uuid.uuid4().hex
    local_id3 = uuid.uuid4().hex
    local_id4 = uuid.uuid4().hex
    local_id5 = uuid.uuid4().hex
    local_entity1 = {'domain_id': self.domainA['id'], 'local_id': local_id1, 'entity_type': mapping.EntityType.USER}
    local_entity2 = {'domain_id': self.domainA['id'], 'local_id': local_id2, 'entity_type': mapping.EntityType.USER}
    local_entity3 = {'domain_id': self.domainB['id'], 'local_id': local_id3, 'entity_type': mapping.EntityType.GROUP}
    local_entity4 = {'domain_id': self.domainB['id'], 'local_id': local_id4, 'entity_type': mapping.EntityType.USER}
    local_entity5 = {'domain_id': self.domainB['id'], 'local_id': local_id5, 'entity_type': mapping.EntityType.USER}
    PROVIDERS.id_mapping_api.create_id_mapping(local_entity1)
    PROVIDERS.id_mapping_api.create_id_mapping(local_entity2)
    PROVIDERS.id_mapping_api.create_id_mapping(local_entity3)
    PROVIDERS.id_mapping_api.create_id_mapping(local_entity4)
    PROVIDERS.id_mapping_api.create_id_mapping(local_entity5)
    PROVIDERS.id_mapping_api.purge_mappings({'domain_id': self.domainA['id']})
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity1))
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity2))
    PROVIDERS.id_mapping_api.purge_mappings({'entity_type': mapping.EntityType.GROUP})
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity3))
    PROVIDERS.id_mapping_api.purge_mappings({'domain_id': self.domainB['id'], 'local_id': local_id4, 'entity_type': mapping.EntityType.USER})
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity4))
    PROVIDERS.id_mapping_api.purge_mappings({})
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity5))