from .segmented import SegmentedStream
def put_read_segment(self, data):
    """
        :param data:
        :type data: bytes
        :return:
        :rtype:
        """
    self._readqueue.put(data)