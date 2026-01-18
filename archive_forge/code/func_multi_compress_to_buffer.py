from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def multi_compress_to_buffer(self, data, threads=-1):
    """
        Compress multiple pieces of data as a single function call.

        (Experimental. Not yet supported by CFFI backend.)

        This function is optimized to perform multiple compression operations
        as as possible with as little overhead as possible.

        Data to be compressed can be passed as a ``BufferWithSegmentsCollection``,
        a ``BufferWithSegments``, or a list containing byte like objects. Each
        element of the container will be compressed individually using the
        configured parameters on the ``ZstdCompressor`` instance.

        The ``threads`` argument controls how many threads to use for
        compression. The default is ``0`` which means to use a single thread.
        Negative values use the number of logical CPUs in the machine.

        The function returns a ``BufferWithSegmentsCollection``. This type
        represents N discrete memory allocations, each holding 1 or more
        compressed frames.

        Output data is written to shared memory buffers. This means that unlike
        regular Python objects, a reference to *any* object within the collection
        keeps the shared buffer and therefore memory backing it alive. This can
        have undesirable effects on process memory usage.

        The API and behavior of this function is experimental and will likely
        change. Known deficiencies include:

        * If asked to use multiple threads, it will always spawn that many
          threads, even if the input is too small to use them. It should
          automatically lower the thread count when the extra threads would
          just add overhead.
        * The buffer allocation strategy is fixed. There is room to make it
          dynamic, perhaps even to allow one output buffer per input,
          facilitating a variation of the API to return a list without the
          adverse effects of shared memory buffers.

        :param data:
           Source to read discrete pieces of data to compress.

           Can be a ``BufferWithSegmentsCollection``, a ``BufferWithSegments``,
           or a ``list[bytes]``.
        :return:
           BufferWithSegmentsCollection holding compressed data.
        """
    raise NotImplementedError()