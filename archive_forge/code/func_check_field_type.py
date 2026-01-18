from django.core import checks
from django.db.backends.base.validation import BaseDatabaseValidation
from django.utils.version import get_docs_version
def check_field_type(self, field, field_type):
    """
        MySQL has the following field length restriction:
        No character (varchar) fields can have a length exceeding 255
        characters if they have a unique index on them.
        MySQL doesn't support a database index on some data types.
        """
    errors = []
    if field_type.startswith('varchar') and field.unique and (field.max_length is None or int(field.max_length) > 255):
        errors.append(checks.Warning('%s may not allow unique CharFields to have a max_length > 255.' % self.connection.display_name, obj=field, hint='See: https://docs.djangoproject.com/en/%s/ref/databases/#mysql-character-fields' % get_docs_version(), id='mysql.W003'))
    if field.db_index and field_type.lower() in self.connection._limited_data_types:
        errors.append(checks.Warning('%s does not support a database index on %s columns.' % (self.connection.display_name, field_type), hint="An index won't be created. Silence this warning if you don't care about it.", obj=field, id='fields.W162'))
    return errors