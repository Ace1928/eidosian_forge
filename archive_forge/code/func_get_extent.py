from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_extent(self, filename, locations):
    """Obtain a SourceRange from this translation unit.

        The bounds of the SourceRange must ultimately be defined by a start and
        end SourceLocation. For the locations argument, you can pass:

          - 2 SourceLocation instances in a 2-tuple or list.
          - 2 int file offsets via a 2-tuple or list.
          - 2 2-tuple or lists of (line, column) pairs in a 2-tuple or list.

        e.g.

        get_extent('foo.c', (5, 10))
        get_extent('foo.c', ((1, 1), (1, 15)))
        """
    f = self.get_file(filename)
    if len(locations) < 2:
        raise Exception('Must pass object with at least 2 elements')
    start_location, end_location = locations
    if hasattr(start_location, '__len__'):
        start_location = SourceLocation.from_position(self, f, start_location[0], start_location[1])
    elif isinstance(start_location, int):
        start_location = SourceLocation.from_offset(self, f, start_location)
    if hasattr(end_location, '__len__'):
        end_location = SourceLocation.from_position(self, f, end_location[0], end_location[1])
    elif isinstance(end_location, int):
        end_location = SourceLocation.from_offset(self, f, end_location)
    assert isinstance(start_location, SourceLocation)
    assert isinstance(end_location, SourceLocation)
    return SourceRange.from_locations(start_location, end_location)