import re
import math
def add_raw_token(self, token):
    for _ in range(token.newlines):
        self.__add_outputline()
    self.current_line.set_indent(-1)
    self.current_line.push(token.whitespace_before)
    self.current_line.push(token.text)
    self.space_before_token = False
    self.non_breaking_space = False
    self.previous_token_wrapped = False