import re
import math
def remove_indent(self, index):
    while index < len(self.__lines):
        self.__lines[index]._remove_indent()
        index += 1
    self.current_line._remove_wrap_indent()