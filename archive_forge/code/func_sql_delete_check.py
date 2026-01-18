from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
@property
def sql_delete_check(self):
    if self.connection.mysql_is_mariadb:
        return 'ALTER TABLE %(table)s DROP CONSTRAINT IF EXISTS %(name)s'
    return 'ALTER TABLE %(table)s DROP CHECK %(name)s'