from __future__ import absolute_import, division, print_function
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
def parse_key_id(key_id):
    """validate the key_id and break it into segments

    :arg key_id: The key_id as supplied by the user.  A valid key_id will be
        8, 16, or more hexadecimal chars with an optional leading ``0x``.
    :returns: The portion of key_id suitable for apt-key del, the portion
        suitable for comparisons with --list-public-keys, and the portion that
        can be used with --recv-key.  If key_id is long enough, these will be
        the last 8 characters of key_id, the last 16 characters, and all of
        key_id.  If key_id is not long enough, some of the values will be the
        same.

    * apt-key del <= 1.10 has a bug with key_id != 8 chars
    * apt-key adv --list-public-keys prints 16 chars
    * apt-key adv --recv-key can take more chars

    """
    int(to_native(key_id), 16)
    key_id = key_id.upper()
    if key_id.startswith('0X'):
        key_id = key_id[2:]
    key_id_len = len(key_id)
    if (key_id_len != 8 and key_id_len != 16) and key_id_len <= 16:
        raise ValueError('key_id must be 8, 16, or 16+ hexadecimal characters in length')
    short_key_id = key_id[-8:]
    fingerprint = key_id
    if key_id_len > 16:
        fingerprint = key_id[-16:]
    return (short_key_id, fingerprint, key_id)