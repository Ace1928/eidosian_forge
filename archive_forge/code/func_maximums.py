import math
import itertools
import random
def maximums(self):
    """Returns all visible maximums value and position sorted with the
        global maximum first.
        """
    maximums = list()
    for func, pos, height, width in zip(self.peaks_function, self.peaks_position, self.peaks_height, self.peaks_width):
        val = func(pos, pos, height, width)
        if val >= self.__call__(pos, count=False)[0]:
            maximums.append((val, pos))
    return sorted(maximums, reverse=True)