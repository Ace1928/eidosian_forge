def server_path_to_key(self, path):
    if path.startswith('config.'):
        return {'section': 'config', 'name': path.split('config.', 1)[1]}
    elif path.startswith('summary_metrics.'):
        return {'section': 'summary', 'name': path.split('summary_metrics.', 1)[1]}
    elif path.startswith('keys_info.keys.'):
        return {'section': 'keys_info', 'name': path.split('keys_info.keys.', 1)[1]}
    elif path.startswith('tags.'):
        return {'section': 'tags', 'name': path.split('tags.', 1)[1]}
    else:
        return {'section': 'run', 'name': path}