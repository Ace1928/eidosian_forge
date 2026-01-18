import re
import winreg
def binipdisplay(s):
    """convert a binary array of ip adresses to a python list"""
    if len(s) % 4 != 0:
        raise EnvironmentError
    ol = []
    for i in range(len(s) / 4):
        s1 = s[:4]
        s = s[4:]
        ip = []
        for j in s1:
            ip.append(str(ord(j)))
        ol.append('.'.join(ip))
    return ol