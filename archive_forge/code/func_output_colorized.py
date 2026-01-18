import ctypes
import logging
import os
def output_colorized(self, message):
    """
            Output a colorized message.

            On Linux and Mac OS X, this method just writes the
            already-colorized message to the stream, since on these
            platforms console streams accept ANSI escape sequences
            for colorization. On Windows, this handler implements a
            subset of ANSI escape sequence handling by parsing the
            message, extracting the sequences and making Win32 API
            calls to colorize the output.

            :param message: The message to colorize and output.
            """
    parts = self.ansi_esc.split(message)
    write = self.stream.write
    h = None
    fd = getattr(self.stream, 'fileno', None)
    if fd is not None:
        fd = fd()
        if fd in (1, 2):
            h = ctypes.windll.kernel32.GetStdHandle(-10 - fd)
    while parts:
        text = parts.pop(0)
        if text:
            write(text)
        if parts:
            params = parts.pop(0)
            if h is not None:
                params = [int(p) for p in params.split(';')]
                color = 0
                for p in params:
                    if 40 <= p <= 47:
                        color |= self.nt_color_map[p - 40] << 4
                    elif 30 <= p <= 37:
                        color |= self.nt_color_map[p - 30]
                    elif p == 1:
                        color |= 8
                    elif p == 0:
                        color = 7
                    else:
                        pass
                ctypes.windll.kernel32.SetConsoleTextAttribute(h, color)