from __future__ import (absolute_import, division, print_function)
import re
def is_same_ip_address(current_ip, applied_ip):
    """
    current_ip can be either an ip of type str or ip and subnet of tye list
    ip like "10.10.10.0"
    ip with subnet mask: ["10.10.10.0", "255.255.255.0"]

    applied_ip can be in 3 formats:
    2 same as above and
    "10.10.10.0/24"
    """
    if isinstance(current_ip, list):
        current_ip = ' '.join(current_ip)
    if len(current_ip) == 0 and len(applied_ip) == 0:
        return True
    if len(current_ip) == 0 or len(applied_ip) == 0:
        return False
    if ' ' not in applied_ip and '/' not in applied_ip:
        return current_ip == applied_ip
    splitted_current_ip = [current_ip]
    splitted_applied_ip = [applied_ip]
    total_bits_current_ip = 0
    total_bits_applied_ip = 0
    if ' ' in current_ip:
        splitted_current_ip = current_ip.split(' ')
    elif '/' in current_ip:
        splitted_current_ip = current_ip.split('/')
    if ' ' in applied_ip:
        splitted_applied_ip = applied_ip.split(' ')
    elif '/' in applied_ip:
        splitted_applied_ip = applied_ip.split('/')
    if splitted_current_ip[0] != splitted_applied_ip[0]:
        return False
    else:
        if '.' in splitted_current_ip[1]:
            total_bits_current_ip = sum([bits(int(s)) for s in splitted_current_ip[1].split('.')])
        else:
            total_bits_current_ip = int(splitted_current_ip[1])
        if '.' in splitted_applied_ip[1]:
            total_bits_applied_ip = sum([bits(int(s)) for s in splitted_applied_ip[1].split('.')])
        else:
            total_bits_applied_ip = int(splitted_applied_ip[1])
        return total_bits_current_ip == total_bits_applied_ip