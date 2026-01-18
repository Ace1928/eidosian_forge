from .error import VerificationError
def format_four_bytes(num):
    return '\\x%02X\\x%02X\\x%02X\\x%02X' % (num >> 24 & 255, num >> 16 & 255, num >> 8 & 255, num & 255)