import os
import sysconfig
def reset_tzpath(to=None):
    global TZPATH
    tzpaths = to
    if tzpaths is not None:
        if isinstance(tzpaths, (str, bytes)):
            raise TypeError(f'tzpaths must be a list or tuple, ' + f'not {type(tzpaths)}: {tzpaths!r}')
        if not all(map(os.path.isabs, tzpaths)):
            raise ValueError(_get_invalid_paths_message(tzpaths))
        base_tzpath = tzpaths
    else:
        env_var = os.environ.get('PYTHONTZPATH', None)
        if env_var is not None:
            base_tzpath = _parse_python_tzpath(env_var)
        else:
            base_tzpath = _parse_python_tzpath(sysconfig.get_config_var('TZPATH'))
    TZPATH = tuple(base_tzpath)