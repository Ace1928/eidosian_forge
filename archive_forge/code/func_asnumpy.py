import cupy
import numpy
def asnumpy(self):
    chunks = [cupy.asnumpy(c) for c in self._chunks.values()]
    return numpy.concatenate(chunks, axis=self._axis)