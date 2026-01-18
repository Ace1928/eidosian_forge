import os
import string
import sys
def get_character():
    """Gets a character from the keyboard and returns the key code"""
    char = get_raw_chars()
    if ord(char) in [KEYMAP['interrupt'], KEYMAP['newline']]:
        return char
    elif ord(char) == KEYMAP['esc']:
        combo = get_raw_chars()
        if ord(combo) == KEYMAP['mod_int']:
            key = get_raw_chars()
            if ord(key) >= KEYMAP['arrow_begin'] - ARROW_KEY_FLAG and ord(key) <= KEYMAP['arrow_end'] - ARROW_KEY_FLAG:
                return chr(ord(key) + ARROW_KEY_FLAG)
            else:
                return KEYMAP['undefined']
        else:
            return get_raw_chars()
    elif char in string.printable:
        return char
    else:
        return KEYMAP['undefined']