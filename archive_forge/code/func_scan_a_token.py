from __future__ import absolute_import
import cython
from . import Errors
from .Regexps import BOL, EOL, EOF
def scan_a_token(self):
    """
        Read the next input sequence recognised by the machine
        and return (text, action). Returns ('', None) on end of
        file.
        """
    self.start_pos = self.cur_pos
    self.current_scanner_position_tuple = (self.name, self.cur_line, self.cur_pos - self.cur_line_start)
    action = self.run_machine_inlined()
    if action is not None:
        if self.trace:
            print('Scanner: read: Performing %s %d:%d' % (action, self.start_pos, self.cur_pos))
        text = self.buffer[self.start_pos - self.buf_start_pos:self.cur_pos - self.buf_start_pos]
        return (text, action)
    else:
        if self.cur_pos == self.start_pos:
            if self.cur_char is EOL:
                self.next_char()
            if self.cur_char is None or self.cur_char is EOF:
                return (u'', None)
        raise Errors.UnrecognizedInput(self, self.state_name)