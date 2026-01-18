from __future__ import absolute_import
import binascii
import codecs
import os
from io import BytesIO
from .fields import RequestField
from .packages import six
from .packages.six import b
def encode_multipart_formdata(fields, boundary=None):
    """
    Encode a dictionary of ``fields`` using the multipart/form-data MIME format.

    :param fields:
        Dictionary of fields or list of (key, :class:`~urllib3.fields.RequestField`).

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`urllib3.filepost.choose_boundary`.
    """
    body = BytesIO()
    if boundary is None:
        boundary = choose_boundary()
    for field in iter_field_objects(fields):
        body.write(b('--%s\r\n' % boundary))
        writer(body).write(field.render_headers())
        data = field.data
        if isinstance(data, int):
            data = str(data)
        if isinstance(data, six.text_type):
            writer(body).write(data)
        else:
            body.write(data)
        body.write(b'\r\n')
    body.write(b('--%s--\r\n' % boundary))
    content_type = str('multipart/form-data; boundary=%s' % boundary)
    return (body.getvalue(), content_type)