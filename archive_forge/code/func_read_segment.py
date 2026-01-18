from .segmented import SegmentedStream
def read_segment(self):
    """
        :return:
        :rtype: bytes
        """
    return self._read.pop(0)