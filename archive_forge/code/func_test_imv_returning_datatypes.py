from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
@testing.combinations(('non_native_uuid', Uuid(native_uuid=False), uuid.uuid4()), ('non_native_uuid_str', Uuid(as_uuid=False, native_uuid=False), str(uuid.uuid4())), ('generic_native_uuid', Uuid(native_uuid=True), uuid.uuid4(), testing.requires.uuid_data_type), ('generic_native_uuid_str', Uuid(as_uuid=False, native_uuid=True), str(uuid.uuid4()), testing.requires.uuid_data_type), ('UUID', UUID(), uuid.uuid4(), testing.requires.uuid_data_type), ('LargeBinary1', LargeBinary(), b'this is binary'), ('LargeBinary2', LargeBinary(), b'7\xe7\x9f'), argnames='type_,value', id_='iaa')
@testing.variation('sort_by_parameter_order', [True, False])
@testing.variation('multiple_rows', [True, False])
@testing.requires.insert_returning
def test_imv_returning_datatypes(self, connection, metadata, sort_by_parameter_order, type_, value, multiple_rows):
    """test #9739, #9808 (similar to #9701).

        this tests insertmanyvalues in conjunction with various datatypes.

        These tests are particularly for the asyncpg driver which needs
        most types to be explicitly cast for the new IMV format

        """
    t = Table('d_t', metadata, Column('id', Integer, Identity(), primary_key=True), Column('value', type_))
    t.create(connection)
    result = connection.execute(t.insert().returning(t.c.id, t.c.value, sort_by_parameter_order=bool(sort_by_parameter_order)), [{'value': value} for i in range(10)] if multiple_rows else {'value': value})
    if multiple_rows:
        i_range = range(1, 11)
    else:
        i_range = range(1, 2)
    eq_(set(result), {(id_, value) for id_ in i_range})
    eq_(set(connection.scalars(select(t.c.value))), {value})