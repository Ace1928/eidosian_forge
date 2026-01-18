import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def get_while(self, test_set, include=True):
    string = self.string
    pos = self.pos
    if self.ignore_space:
        try:
            substring = []
            while True:
                if string[pos].isspace():
                    pos += 1
                elif string[pos] == '#':
                    pos = string.index('\n', pos)
                elif (string[pos] in test_set) == include:
                    substring.append(string[pos])
                    pos += 1
                else:
                    break
            self.pos = pos
        except IndexError:
            self.pos = len(string)
        except ValueError:
            self.pos = len(string)
        return ''.join(substring)
    else:
        try:
            while (string[pos] in test_set) == include:
                pos += 1
            substring = string[self.pos:pos]
            self.pos = pos
            return substring
        except IndexError:
            substring = string[self.pos:pos]
            self.pos = pos
            return substring