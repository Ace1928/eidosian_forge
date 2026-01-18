from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def skip_default_on_alter(self, field):
    if self._is_limited_data_type(field) and (not self.connection.mysql_is_mariadb):
        return True
    return False