from typing import List, Optional
def getspnam(self, username: str) -> _ShadowRecord:
    """
        Return the shadow user record corresponding to the given username.
        """
    if not isinstance(username, str):
        raise TypeError(f'getspnam() argument must be str, not {type(username)}')
    for entry in self._users:
        if entry.sp_nam == username:
            return entry
    raise KeyError(username)