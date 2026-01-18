import re
import itertools
def setupTimingPattern(self):
    for r in range(8, self.moduleCount - 8):
        self.modules[r][6] = r % 2 == 0
    self.modules[6][8:self.moduleCount - 8] = itertools.islice(itertools.cycle([True, False]), self.moduleCount - 16)