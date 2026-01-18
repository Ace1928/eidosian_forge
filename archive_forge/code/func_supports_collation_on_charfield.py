from django.db import DatabaseError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_collation_on_charfield(self):
    with self.connection.cursor() as cursor:
        try:
            cursor.execute("SELECT CAST('a' AS VARCHAR2(4001)) FROM dual")
        except DatabaseError as e:
            if e.args[0].code == 910:
                return False
            raise
        return True