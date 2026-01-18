def key_to_server_path(self, key):
    if key['section'] == 'config':
        return 'config.' + key['name']
    elif key['section'] == 'summary':
        return 'summary_metrics.' + key['name']
    elif key['section'] == 'keys_info':
        return 'keys_info.keys.' + key['name']
    elif key['section'] == 'run':
        return key['name']
    elif key['section'] == 'tags':
        return 'tags.' + key['name']
    raise ValueError('Invalid key: %s' % key)