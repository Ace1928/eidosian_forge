from winappdbg.textio import HexInput
from winappdbg.util import StaticClass, MemoryAddresses
from winappdbg import win32
import warnings
@classmethod
def extract_ascii_strings(cls, process, minSize=4, maxSize=1024):
    """
        Extract ASCII strings from the process memory.

        @type  process: L{Process}
        @param process: Process to search.

        @type  minSize: int
        @param minSize: (Optional) Minimum size of the strings to search for.

        @type  maxSize: int
        @param maxSize: (Optional) Maximum size of the strings to search for.

        @rtype:  iterator of tuple(int, int, str)
        @return: Iterator of strings extracted from the process memory.
            Each tuple contains the following:
             - The memory address where the string was found.
             - The size of the string.
             - The string.
        """
    regexp = '[\\s\\w\\!\\@\\#\\$\\%%\\^\\&\\*\\(\\)\\{\\}\\[\\]\\~\\`\\\'\\"\\:\\;\\.\\,\\\\\\/\\-\\+\\=\\_\\<\\>]{%d,%d}\\0' % (minSize, maxSize)
    pattern = RegExpPattern(regexp, 0, maxSize)
    return cls.search_process(process, pattern, overlapping=False)