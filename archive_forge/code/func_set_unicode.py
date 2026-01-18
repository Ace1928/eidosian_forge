import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def set_unicode(self, enabled_flag):
    """Windows: if 'enabled_flag' is True, enable the UNICODE and
        _UNICODE defines in C, and declare the types like TCHAR and LPTCSTR
        to be (pointers to) wchar_t.  If 'enabled_flag' is False,
        declare these types to be (pointers to) plain 8-bit characters.
        This is mostly for backward compatibility; you usually want True.
        """
    if self._windows_unicode is not None:
        raise ValueError('set_unicode() can only be called once')
    enabled_flag = bool(enabled_flag)
    if enabled_flag:
        self.cdef('typedef wchar_t TBYTE;typedef wchar_t TCHAR;typedef const wchar_t *LPCTSTR;typedef const wchar_t *PCTSTR;typedef wchar_t *LPTSTR;typedef wchar_t *PTSTR;typedef TBYTE *PTBYTE;typedef TCHAR *PTCHAR;')
    else:
        self.cdef('typedef char TBYTE;typedef char TCHAR;typedef const char *LPCTSTR;typedef const char *PCTSTR;typedef char *LPTSTR;typedef char *PTSTR;typedef TBYTE *PTBYTE;typedef TCHAR *PTCHAR;')
    self._windows_unicode = enabled_flag