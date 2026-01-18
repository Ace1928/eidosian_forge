import struct
from ..common.utils import struct_parse
from .sections import Section
def get_number_of_symbols(self):
    """ Get the number of symbols in the hash table by finding the bucket
            with the highest symbol index and walking to the end of its chain.
        """
    max_idx = max(self.params['buckets'])
    if max_idx < self.params['symoffset']:
        return self.params['symoffset']
    max_chain_pos = self._chain_pos + (max_idx - self.params['symoffset']) * self._wordsize
    self.elffile.stream.seek(max_chain_pos)
    hash_format = '<I' if self.elffile.little_endian else '>I'
    while True:
        cur_hash = struct.unpack(hash_format, self.elffile.stream.read(self._wordsize))[0]
        if cur_hash & 1:
            return max_idx + 1
        max_idx += 1