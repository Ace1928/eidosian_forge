import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def iter_attachments(self):
    """Return an iterator over the non-main parts of a multipart.

        Skip the first of each occurrence of text/plain, text/html,
        multipart/related, or multipart/alternative in the multipart (unless
        they have a 'Content-Disposition: attachment' header) and include all
        remaining subparts in the returned iterator.  When applied to a
        multipart/related, return all parts except the root part.  Return an
        empty iterator when applied to a multipart/alternative or a
        non-multipart.
        """
    maintype, subtype = self.get_content_type().split('/')
    if maintype != 'multipart' or subtype == 'alternative':
        return
    payload = self.get_payload()
    try:
        parts = payload.copy()
    except AttributeError:
        return
    if maintype == 'multipart' and subtype == 'related':
        start = self.get_param('start')
        if start:
            found = False
            attachments = []
            for part in parts:
                if part.get('content-id') == start:
                    found = True
                else:
                    attachments.append(part)
            if found:
                yield from attachments
                return
        parts.pop(0)
        yield from parts
        return
    seen = []
    for part in parts:
        maintype, subtype = part.get_content_type().split('/')
        if (maintype, subtype) in self._body_types and (not part.is_attachment()) and (subtype not in seen):
            seen.append(subtype)
            continue
        yield part