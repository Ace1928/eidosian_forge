from winappdbg.textio import HexInput
from winappdbg.util import StaticClass, MemoryAddresses
from winappdbg import win32
import warnings
@staticmethod
def search_process(process, pattern, minAddr=None, maxAddr=None, bufferPages=None, overlapping=False):
    """
        Search for the given pattern within the process memory.

        @type  process: L{Process}
        @param process: Process to search.

        @type  pattern: L{Pattern}
        @param pattern: Pattern to search for.
            It must be an instance of a subclass of L{Pattern}.

            The following L{Pattern} subclasses are provided by WinAppDbg:
             - L{BytePattern}
             - L{TextPattern}
             - L{RegExpPattern}
             - L{HexPattern}

            You can also write your own subclass of L{Pattern} for customized
            searches.

        @type  minAddr: int
        @param minAddr: (Optional) Start the search at this memory address.

        @type  maxAddr: int
        @param maxAddr: (Optional) Stop the search at this memory address.

        @type  bufferPages: int
        @param bufferPages: (Optional) Number of memory pages to buffer when
            performing the search. Valid values are:
             - C{0} or C{None}:
               Automatically determine the required buffer size. May not give
               complete results for regular expressions that match variable
               sized strings.
             - C{> 0}: Set the buffer size, in memory pages.
             - C{< 0}: Disable buffering entirely. This may give you a little
               speed gain at the cost of an increased memory usage. If the
               target process has very large contiguous memory regions it may
               actually be slower or even fail. It's also the only way to
               guarantee complete results for regular expressions that match
               variable sized strings.

        @type  overlapping: bool
        @param overlapping: C{True} to allow overlapping results, C{False}
            otherwise.

            Overlapping results yield the maximum possible number of results.

            For example, if searching for "AAAA" within "AAAAAAAA" at address
            C{0x10000}, when overlapping is turned off the following matches
            are yielded::
                (0x10000, 4, "AAAA")
                (0x10004, 4, "AAAA")

            If overlapping is turned on, the following matches are yielded::
                (0x10000, 4, "AAAA")
                (0x10001, 4, "AAAA")
                (0x10002, 4, "AAAA")
                (0x10003, 4, "AAAA")
                (0x10004, 4, "AAAA")

            As you can see, the middle results are overlapping the last two.

        @rtype:  iterator of tuple( int, int, str )
        @return: An iterator of tuples. Each tuple contains the following:
             - The memory address where the pattern was found.
             - The size of the data that matches the pattern.
             - The data that matches the pattern.

        @raise WindowsError: An error occurred when querying or reading the
            process memory.
        """
    MEM_COMMIT = win32.MEM_COMMIT
    PAGE_GUARD = win32.PAGE_GUARD
    page = MemoryAddresses.pageSize
    read = pattern.read
    find = pattern.find
    if minAddr is None:
        minAddr = 0
    if maxAddr is None:
        maxAddr = win32.LPVOID(-1).value
    if bufferPages is None:
        try:
            size = MemoryAddresses.align_address_to_page_end(len(pattern)) + page
        except NotImplementedError:
            size = None
    elif bufferPages > 0:
        size = page * (bufferPages + 1)
    else:
        size = None
    memory_map = process.iter_memory_map(minAddr, maxAddr)
    if size:
        buffer = ''
        prev_addr = 0
        last = 0
        delta = 0
        for mbi in memory_map:
            if not mbi.has_content():
                continue
            address = mbi.BaseAddress
            block_size = mbi.RegionSize
            if address >= maxAddr:
                break
            end = address + block_size
            if delta and address == prev_addr:
                buffer += read(process, address, page)
            else:
                buffer = read(process, address, min(size, block_size))
                last = 0
                delta = 0
            while 1:
                pos, length = find(buffer, last)
                while pos >= last:
                    match_addr = address + pos - delta
                    if minAddr <= match_addr < maxAddr:
                        result = pattern.found(match_addr, length, buffer[pos:pos + length])
                        if result is not None:
                            yield result
                    if overlapping:
                        last = pos + 1
                    else:
                        last = pos + length
                    pos, length = find(buffer, last)
                address = address + page
                block_size = block_size - page
                prev_addr = address
                last = last - page
                if last < 0:
                    last = 0
                buffer = buffer[page:]
                delta = page
                if address < end:
                    buffer = buffer + read(process, address, page)
                else:
                    break
    else:
        for mbi in memory_map:
            if not mbi.has_content():
                continue
            address = mbi.BaseAddress
            block_size = mbi.RegionSize
            if address >= maxAddr:
                break
            buffer = process.read(address, block_size)
            pos, length = find(buffer)
            last = 0
            while pos >= last:
                match_addr = address + pos
                if minAddr <= match_addr < maxAddr:
                    result = pattern.found(match_addr, length, buffer[pos:pos + length])
                    if result is not None:
                        yield result
                if overlapping:
                    last = pos + 1
                else:
                    last = pos + length
                pos, length = find(buffer, last)