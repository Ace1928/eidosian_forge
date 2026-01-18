import struct
def get_abbr(idx):
    if idx not in abbr_vals:
        span_end = abbr_chars.find(b'\x00', idx)
        abbr_vals[idx] = abbr_chars[idx:span_end].decode()
    return abbr_vals[idx]