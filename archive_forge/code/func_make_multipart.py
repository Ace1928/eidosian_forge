from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def make_multipart(self, content_disposition=None, content_type=None, content_location=None):
    """
        Makes this request field into a multipart request field.

        This method overrides "Content-Disposition", "Content-Type" and
        "Content-Location" headers to the request parameter.

        :param content_type:
            The 'Content-Type' of the request body.
        :param content_location:
            The 'Content-Location' of the request body.

        """
    self.headers['Content-Disposition'] = content_disposition or u'form-data'
    self.headers['Content-Disposition'] += u'; '.join([u'', self._render_parts(((u'name', self._name), (u'filename', self._filename)))])
    self.headers['Content-Type'] = content_type
    self.headers['Content-Location'] = content_location