from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
def get_unserialized_body(self):
    """Return a dict suitable for serialization in the wsgi controller.

        This wraps the exception details in a format which maps to the
        expected format for the AWS API.
        """
    if self.detail:
        message = ':'.join([self.explanation, self.detail])
    else:
        message = self.explanation
    return {'ErrorResponse': {'Error': {'Type': self.err_type, 'Code': self.title, 'Message': message}}}