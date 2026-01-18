from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
def get_attributes_for_oid(self, oid: ObjectIdentifier) -> typing.List[NameAttribute]:
    return [i for i in self if i.oid == oid]