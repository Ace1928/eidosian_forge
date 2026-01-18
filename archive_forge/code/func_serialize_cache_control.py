import re
def serialize_cache_control(properties):
    if isinstance(properties, CacheControl):
        properties = properties.properties
    parts = []
    for name, value in sorted(properties.items()):
        if value is None:
            parts.append(name)
            continue
        value = str(value)
        if need_quote_re.search(value):
            value = '"%s"' % value
        parts.append('%s=%s' % (name, value))
    return ', '.join(parts)