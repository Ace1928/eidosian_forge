import binascii
import warnings
from webob.compat import (
@classmethod
def from_fieldstorage(cls, fs):
    """
        Create a dict from a cgi.FieldStorage instance
        """
    obj = cls()
    for field in fs.list or ():
        charset = field.type_options.get('charset', 'utf8')
        transfer_encoding = field.headers.get('Content-Transfer-Encoding', None)
        supported_transfer_encoding = {'base64': binascii.a2b_base64, 'quoted-printable': binascii.a2b_qp}
        if not PY2:
            if charset == 'utf8':
                decode = lambda b: b
            else:
                decode = lambda b: b.encode('utf8').decode(charset)
        else:
            decode = lambda b: b.decode(charset)
        if field.filename:
            field.filename = decode(field.filename)
            obj.add(field.name, field)
        else:
            value = field.value
            if transfer_encoding in supported_transfer_encoding:
                if not PY2:
                    value = value.encode('utf8')
                value = supported_transfer_encoding[transfer_encoding](value)
                if not PY2:
                    value = value.decode('utf8')
            obj.add(field.name, decode(value))
    return obj