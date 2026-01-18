import os
def ensure_writable_plotly_dir():
    global _file_permissions
    if _file_permissions is None:
        _file_permissions = _permissions()
    return _file_permissions