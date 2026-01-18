import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
def set_generic_password(name, service, username, password):
    with contextlib.suppress(NotFound):
        delete_generic_password(name, service, username)
    q = create_query(kSecClass=k_('kSecClassGenericPassword'), kSecAttrService=service, kSecAttrAccount=username, kSecValueData=password)
    status = SecItemAdd(q, None)
    Error.raise_for_status(status)