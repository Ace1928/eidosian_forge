def normalize_keys(config):
    new_config = {}
    for key, value in config.items():
        key = key.replace('-', '_')
        if isinstance(value, dict):
            new_config[key] = normalize_keys(value)
        elif isinstance(value, bool):
            new_config[key] = value
        elif isinstance(value, int) and key not in ('verbose_level', 'api_timeout'):
            new_config[key] = str(value)
        elif isinstance(value, float):
            new_config[key] = str(value)
        else:
            new_config[key] = value
    return new_config