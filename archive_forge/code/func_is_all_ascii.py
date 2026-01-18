def is_all_ascii(text):
    for c in text:
        if ord(c) > 127:
            return False
    return True