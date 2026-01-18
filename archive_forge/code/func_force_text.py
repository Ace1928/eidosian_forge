def force_text(data, encoding='utf-8'):
    if isinstance(data, str):
        return data
    if isinstance(data, bytes):
        return data.decode(encoding)
    return str(data, encoding)