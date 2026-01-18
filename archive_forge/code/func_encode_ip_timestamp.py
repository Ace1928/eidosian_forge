import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def encode_ip_timestamp(ip, timestamp):
    ip_chars = b''.join(map(int2byte, map(int, ip.split('.'))))
    t = int(timestamp)
    ts = ((t & 4278190080) >> 24, (t & 16711680) >> 16, (t & 65280) >> 8, t & 255)
    ts_chars = b''.join(map(int2byte, ts))
    return ip_chars + ts_chars