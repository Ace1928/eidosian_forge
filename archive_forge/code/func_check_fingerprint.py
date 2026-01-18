import re
def check_fingerprint(path):
    path_parts = path.split('/')
    name_parts = path_parts[-1].split('.')
    if len(name_parts) > 2 and cache_regex.match(name_parts[1]):
        original_name = '.'.join([name_parts[0]] + name_parts[2:])
        return ('/'.join(path_parts[:-1] + [original_name]), True)
    return (path, False)