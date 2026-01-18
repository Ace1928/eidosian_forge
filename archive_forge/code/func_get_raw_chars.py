import os
import string
import sys
def get_raw_chars():
    """Gets raw characters from inputs"""
    if os.name == 'nt':
        import msvcrt
        encoding = 'mbcs'
        while msvcrt.kbhit():
            msvcrt.getch()
        if len(WIN_CH_BUFFER) == 0:
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'):
                ch2 = ch + msvcrt.getch()
                try:
                    chx = chr(WIN_KEYMAP[ch2])
                    WIN_CH_BUFFER.append(chr(KEYMAP['mod_int']))
                    WIN_CH_BUFFER.append(chx)
                    if ord(chx) in (KEYMAP['insert'] - 1 << 9, KEYMAP['delete'] - 1 << 9, KEYMAP['pg_up'] - 1 << 9, KEYMAP['pg_down'] - 1 << 9):
                        WIN_CH_BUFFER.append(chr(126))
                    ch = chr(KEYMAP['esc'])
                except KeyError:
                    ch = ch2[1]
            else:
                ch = ch.decode(encoding)
        else:
            ch = WIN_CH_BUFFER.pop(0)
    elif os.name == 'posix':
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch