import posixpath
def shift_path_info(environ):
    """Shift a name from PATH_INFO to SCRIPT_NAME, returning it

    If there are no remaining path segments in PATH_INFO, return None.
    Note: 'environ' is modified in-place; use a copy if you need to keep
    the original PATH_INFO or SCRIPT_NAME.

    Note: when PATH_INFO is just a '/', this returns '' and appends a trailing
    '/' to SCRIPT_NAME, even though empty path segments are normally ignored,
    and SCRIPT_NAME doesn't normally end in a '/'.  This is intentional
    behavior, to ensure that an application can tell the difference between
    '/x' and '/x/' when traversing to objects.
    """
    path_info = environ.get('PATH_INFO', '')
    if not path_info:
        return None
    path_parts = path_info.split('/')
    path_parts[1:-1] = [p for p in path_parts[1:-1] if p and p != '.']
    name = path_parts[1]
    del path_parts[1]
    script_name = environ.get('SCRIPT_NAME', '')
    script_name = posixpath.normpath(script_name + '/' + name)
    if script_name.endswith('/'):
        script_name = script_name[:-1]
    if not name and (not script_name.endswith('/')):
        script_name += '/'
    environ['SCRIPT_NAME'] = script_name
    environ['PATH_INFO'] = '/'.join(path_parts)
    if name == '.':
        name = None
    return name