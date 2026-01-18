import base64
import datetime
import re
import struct
import typing as t
import uuid
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
@per_sequence
def parse_dn(value: str) -> t.List[t.List[str]]:
    """Parses a DistinguishedName and emits a structured object."""
    dn: t.List[t.List[str]] = []
    b_value = value.encode('utf-8', errors='surrogateescape')
    b_view = memoryview(b_value)
    while b_view:
        rdns: t.List[str] = []
        while True:
            attr_type = _parse_rdn_type(b_view)
            if not attr_type:
                remaining = b_view.tobytes().decode('utf-8', errors='surrogateescape')
                raise AnsibleFilterError(f"Expecting attribute type in RDN entry from '{remaining}'")
            rdns.append(attr_type[0].decode('utf-8', errors='surrogateescape'))
            b_view = b_view[attr_type[1]:]
            attr_value = _parse_rdn_value(b_view)
            if not attr_value:
                remaining = b_view.tobytes().decode('utf-8', errors='surrogateescape')
                raise AnsibleFilterError(f"Expecting attribute value in RDN entry from '{remaining}'")
            rdns.append(attr_value[0].decode('utf-8', errors='surrogateescape'))
            b_view = b_view[attr_value[1]:]
            if attr_value[2]:
                continue
            else:
                break
        dn.append(rdns)
    return dn