import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
def test_id_mapping_crud(self):
    initial_mappings = len(mapping_sql.list_id_mappings())
    local_id1 = uuid.uuid4().hex
    local_id2 = uuid.uuid4().hex
    local_entity1 = {'domain_id': self.domainA['id'], 'local_id': local_id1, 'entity_type': mapping.EntityType.USER}
    local_entity2 = {'domain_id': self.domainB['id'], 'local_id': local_id2, 'entity_type': mapping.EntityType.GROUP}
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity1))
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity2))
    public_id1 = PROVIDERS.id_mapping_api.create_id_mapping(local_entity1)
    public_id2 = PROVIDERS.id_mapping_api.create_id_mapping(local_entity2)
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 2))
    self.assertEqual(public_id1, PROVIDERS.id_mapping_api.get_public_id(local_entity1))
    self.assertEqual(public_id2, PROVIDERS.id_mapping_api.get_public_id(local_entity2))
    local_id_ref = PROVIDERS.id_mapping_api.get_id_mapping(public_id1)
    self.assertEqual(self.domainA['id'], local_id_ref['domain_id'])
    self.assertEqual(local_id1, local_id_ref['local_id'])
    self.assertEqual(mapping.EntityType.USER, local_id_ref['entity_type'])
    self.assertNotEqual(local_id1, public_id1)
    local_id_ref = PROVIDERS.id_mapping_api.get_id_mapping(public_id2)
    self.assertEqual(self.domainB['id'], local_id_ref['domain_id'])
    self.assertEqual(local_id2, local_id_ref['local_id'])
    self.assertEqual(mapping.EntityType.GROUP, local_id_ref['entity_type'])
    self.assertNotEqual(local_id2, public_id2)
    new_public_id = uuid.uuid4().hex
    public_id3 = PROVIDERS.id_mapping_api.create_id_mapping({'domain_id': self.domainB['id'], 'local_id': local_id2, 'entity_type': mapping.EntityType.USER}, public_id=new_public_id)
    self.assertEqual(new_public_id, public_id3)
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 3))
    PROVIDERS.id_mapping_api.delete_id_mapping(public_id1)
    PROVIDERS.id_mapping_api.delete_id_mapping(public_id2)
    PROVIDERS.id_mapping_api.delete_id_mapping(public_id3)
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings))