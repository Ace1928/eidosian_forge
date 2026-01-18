import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
def delete_generic_password(name, service, username):
    q = create_query(kSecClass=k_('kSecClassGenericPassword'), kSecAttrService=service, kSecAttrAccount=username)
    status = SecItemDelete(q)
    Error.raise_for_status(status)