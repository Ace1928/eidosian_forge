import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
@classmethod
def raise_for_status(cls, status):
    if status == 0:
        return
    if status == error.item_not_found:
        raise NotFound(status, 'Item not found')
    if status == error.keychain_denied:
        raise KeychainDenied(status, 'Keychain Access Denied')
    if status == error.sec_auth_failed or status == error.plist_missing:
        raise SecAuthFailure(status, 'Security Auth Failure: make sure executable is signed with codesign util')
    raise cls(status, 'Unknown Error')