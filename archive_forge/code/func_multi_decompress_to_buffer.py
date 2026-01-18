from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def multi_decompress_to_buffer(self, frames, decompressed_sizes=None, threads=0):
    """
        Decompress multiple zstd frames to output buffers as a single operation.

        (Experimental. Not available in CFFI backend.)

        Compressed frames can be passed to the function as a
        ``BufferWithSegments``, a ``BufferWithSegmentsCollection``, or as a
        list containing objects that conform to the buffer protocol. For best
        performance, pass a ``BufferWithSegmentsCollection`` or a
        ``BufferWithSegments``, as minimal input validation will be done for
        that type. If calling from Python (as opposed to C), constructing one
        of these instances may add overhead cancelling out the performance
        overhead of validation for list inputs.

        Returns a ``BufferWithSegmentsCollection`` containing the decompressed
        data. All decompressed data is allocated in a single memory buffer. The
        ``BufferWithSegments`` instance tracks which objects are at which offsets
        and their respective lengths.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> results = dctx.multi_decompress_to_buffer([b'...', b'...'])

        The decompressed size of each frame MUST be discoverable. It can either be
        embedded within the zstd frame or passed in via the ``decompressed_sizes``
        argument.

        The ``decompressed_sizes`` argument is an object conforming to the buffer
        protocol which holds an array of 64-bit unsigned integers in the machine's
        native format defining the decompressed sizes of each frame. If this argument
        is passed, it avoids having to scan each frame for its decompressed size.
        This frame scanning can add noticeable overhead in some scenarios.

        >>> frames = [...]
        >>> sizes = struct.pack('=QQQQ', len0, len1, len2, len3)
        >>>
        >>> dctx = zstandard.ZstdDecompressor()
        >>> results = dctx.multi_decompress_to_buffer(frames, decompressed_sizes=sizes)

        .. note::

           It is possible to pass a ``mmap.mmap()`` instance into this function by
           wrapping it with a ``BufferWithSegments`` instance (which will define the
           offsets of frames within the memory mapped region).

        This function is logically equivalent to performing
        :py:meth:`ZstdCompressor.decompress` on each input frame and returning the
        result.

        This function exists to perform decompression on multiple frames as fast
        as possible by having as little overhead as possible. Since decompression is
        performed as a single operation and since the decompressed output is stored in
        a single buffer, extra memory allocations, Python objects, and Python function
        calls are avoided. This is ideal for scenarios where callers know up front that
        they need to access data for multiple frames, such as when  *delta chains* are
        being used.

        Currently, the implementation always spawns multiple threads when requested,
        even if the amount of work to do is small. In the future, it will be smarter
        about avoiding threads and their associated overhead when the amount of
        work to do is small.

        :param frames:
           Source defining zstd frames to decompress.
        :param decompressed_sizes:
           Array of integers representing sizes of decompressed zstd frames.
        :param threads:
           How many threads to use for decompression operations.

           Negative values will use the same number of threads as logical CPUs
           on the machine. Values ``0`` or ``1`` use a single thread.
        :return:
           ``BufferWithSegmentsCollection``
        """
    raise NotImplementedError()