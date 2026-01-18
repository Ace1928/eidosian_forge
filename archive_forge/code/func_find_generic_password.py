import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
def find_generic_password(kc_name, service, username, not_found_ok=False):
    q = create_query(kSecClass=k_('kSecClassGenericPassword'), kSecMatchLimit=k_('kSecMatchLimitOne'), kSecAttrService=service, kSecAttrAccount=username, kSecReturnData=create_cfbool(True))
    data = c_void_p()
    status = SecItemCopyMatching(q, byref(data))
    if status == error.item_not_found and not_found_ok:
        return
    Error.raise_for_status(status)
    return cfstr_to_str(data)