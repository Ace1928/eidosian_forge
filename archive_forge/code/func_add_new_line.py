import re
import math
def add_new_line(self, force_newline=False):
    if self.is_empty() or (not force_newline and self.just_added_newline()):
        return False
    if not self.raw:
        self.__add_outputline()
    return True