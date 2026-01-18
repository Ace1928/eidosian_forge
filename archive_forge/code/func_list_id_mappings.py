from keystone.common import sql
from keystone.identity.mapping_backends import sql as mapping_sql
def list_id_mappings():
    """List all id_mappings for testing purposes."""
    with sql.session_for_read() as session:
        refs = session.query(mapping_sql.IDMapping).all()
        return [x.to_dict() for x in refs]