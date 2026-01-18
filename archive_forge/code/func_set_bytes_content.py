import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def set_bytes_content(msg, data, maintype, subtype, cte='base64', disposition=None, filename=None, cid=None, params=None, headers=None):
    _prepare_set(msg, maintype, subtype, headers)
    if cte == 'base64':
        data = _encode_base64(data, max_line_length=msg.policy.max_line_length)
    elif cte == 'quoted-printable':
        data = binascii.b2a_qp(data, istext=False, header=False, quotetabs=True)
        data = data.decode('ascii')
    elif cte == '7bit':
        data = data.decode('ascii')
    elif cte in ('8bit', 'binary'):
        data = data.decode('ascii', 'surrogateescape')
    msg.set_payload(data)
    msg['Content-Transfer-Encoding'] = cte
    _finalize_set(msg, disposition, filename, cid, params)