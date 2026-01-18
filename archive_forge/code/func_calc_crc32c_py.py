from ._crc32c import crc as crc32c_py
from aiokafka.util import NO_EXTENSIONS
def calc_crc32c_py(memview):
    """ Calculate CRC-32C (Castagnoli) checksum over a memoryview of data
    """
    crc = crc32c_py(memview)
    return crc