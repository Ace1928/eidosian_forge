import re
def build_fingerprint(path, version, hash_value):
    path_parts = path.split('/')
    filename, extension = path_parts[-1].split('.', 1)
    file_path = '/'.join(path_parts[:-1] + [filename])
    v_str = re.sub(version_clean, '_', str(version))
    return f'{file_path}.v{v_str}m{hash_value}.{extension}'