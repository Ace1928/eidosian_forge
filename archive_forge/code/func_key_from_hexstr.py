import time
import hashlib
import pyzor
def key_from_hexstr(s):
    try:
        salt, key = s.split(',')
    except ValueError:
        raise ValueError('Invalid number of parts for key; perhaps you forgot the comma at the beginning for the salt divider?')
    return (salt, key)