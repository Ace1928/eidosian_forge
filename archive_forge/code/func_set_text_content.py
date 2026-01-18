import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def set_text_content(msg, string, subtype='plain', charset='utf-8', cte=None, disposition=None, filename=None, cid=None, params=None, headers=None):
    _prepare_set(msg, 'text', subtype, headers)
    cte, payload = _encode_text(string, charset, cte, msg.policy)
    msg.set_payload(payload)
    msg.set_param('charset', email.charset.ALIASES.get(charset, charset), replace=True)
    msg['Content-Transfer-Encoding'] = cte
    _finalize_set(msg, disposition, filename, cid, params)