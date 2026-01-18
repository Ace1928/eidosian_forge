from __future__ import absolute_import, division, print_function
def get_interface_number(name):
    digits = ''
    for char in name:
        if char.isdigit() or char in '/.':
            digits += char
    return digits