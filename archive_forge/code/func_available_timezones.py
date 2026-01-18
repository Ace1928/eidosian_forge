import os
import sysconfig
def available_timezones():
    """Returns a set containing all available time zones.

    .. caution::

        This may attempt to open a large number of files, since the best way to
        determine if a given file on the time zone search path is to open it
        and check for the "magic string" at the beginning.
    """
    from importlib import resources
    valid_zones = set()
    try:
        with resources.files('tzdata').joinpath('zones').open('r') as f:
            for zone in f:
                zone = zone.strip()
                if zone:
                    valid_zones.add(zone)
    except (ImportError, FileNotFoundError):
        pass

    def valid_key(fpath):
        try:
            with open(fpath, 'rb') as f:
                return f.read(4) == b'TZif'
        except Exception:
            return False
    for tz_root in TZPATH:
        if not os.path.exists(tz_root):
            continue
        for root, dirnames, files in os.walk(tz_root):
            if root == tz_root:
                if 'right' in dirnames:
                    dirnames.remove('right')
                if 'posix' in dirnames:
                    dirnames.remove('posix')
            for file in files:
                fpath = os.path.join(root, file)
                key = os.path.relpath(fpath, start=tz_root)
                if os.sep != '/':
                    key = key.replace(os.sep, '/')
                if not key or key in valid_zones:
                    continue
                if valid_key(fpath):
                    valid_zones.add(key)
    if 'posixrules' in valid_zones:
        valid_zones.remove('posixrules')
    return valid_zones