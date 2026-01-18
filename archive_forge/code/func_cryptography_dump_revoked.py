from __future__ import absolute_import, division, print_function
from .basic import (
from .cryptography_support import (
from ._obj2txt import (
def cryptography_dump_revoked(entry, idn_rewrite='ignore'):
    return {'serial_number': entry['serial_number'], 'revocation_date': entry['revocation_date'].strftime(TIMESTAMP_FORMAT), 'issuer': [cryptography_decode_name(issuer, idn_rewrite=idn_rewrite) for issuer in entry['issuer']] if entry['issuer'] is not None else None, 'issuer_critical': entry['issuer_critical'], 'reason': REVOCATION_REASON_MAP_INVERSE.get(entry['reason']) if entry['reason'] is not None else None, 'reason_critical': entry['reason_critical'], 'invalidity_date': entry['invalidity_date'].strftime(TIMESTAMP_FORMAT) if entry['invalidity_date'] is not None else None, 'invalidity_date_critical': entry['invalidity_date_critical']}