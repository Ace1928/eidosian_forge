def get_char_metrics(self, char_num, fix_rsfs=False):
    """Return glyph metrics for a unicode code point.

        Arguments:
            char_num: a unicode code point
            fix_rsfs: adjust for rsfs10.tfm's different indexing system
        """
    if char_num < self.start_char or char_num > self.end_char:
        raise RuntimeError('Invalid character number')
    if fix_rsfs:
        info = self.char_info[char_num - self.start_char]
    else:
        info = self.char_info[char_num + self.start_char]
    char_kern_table = {}
    if info.has_ligkern():
        for char in range(self.start_char, self.end_char + 1):
            kern = self.ligkern_program.execute(info.ligkern_start(), char)
            if kern:
                char_kern_table[char] = self.kern_table[kern]
    return TfmCharMetrics(self.width_table[info.width_index], self.height_table[info.height_index], self.depth_table[info.depth_index], self.italic_table[info.italic_index], char_kern_table)