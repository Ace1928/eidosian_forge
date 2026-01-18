import re
def readIgnored(self, input):
    return input.readUntilAfter(self.__directives_end_ignore_pattern)