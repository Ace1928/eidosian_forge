import re
import math
def ensure_empty_line_above(self, starts_with, ends_with):
    index = len(self.__lines) - 2
    while index >= 0:
        potentialEmptyLine = self.__lines[index]
        if potentialEmptyLine.is_empty():
            break
        elif not potentialEmptyLine.item(0).startswith(starts_with) and potentialEmptyLine.item(-1) != ends_with:
            self.__lines.insert(index + 1, OutputLine(self))
            self.previous_line = self.__lines[-2]
            break
        index -= 1