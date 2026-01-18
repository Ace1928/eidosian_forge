from .. import osutils
def make_delta(self, new_lines, bytes_length, soft=False):
    """Compute the delta for this content versus the original content."""
    out_lines = [b'', b'', encode_base128_int(bytes_length)]
    index_lines = [False, False, False]
    output_handler = _OutputHandler(out_lines, index_lines, self._MIN_MATCH_BYTES)
    blocks = self.get_matching_blocks(new_lines, soft=soft)
    current_line_num = 0
    for old_start, new_start, range_len in blocks:
        if new_start != current_line_num:
            output_handler.add_insert(new_lines[current_line_num:new_start])
        current_line_num = new_start + range_len
        if range_len:
            if old_start == 0:
                first_byte = 0
            else:
                first_byte = self.line_offsets[old_start - 1]
            last_byte = self.line_offsets[old_start + range_len - 1]
            output_handler.add_copy(first_byte, last_byte)
    return (out_lines, index_lines)