from Bio.PopGen.GenePop import get_indiv
def remove_loci_by_position(self, positions, fname):
    """Remove a set of loci by position.

        Arguments:
         - positions - positions
         - fname - file to be created with locus removed

        """
    old_rec = read(self.fname)
    with open(fname, 'w') as f:
        f.write(self.comment_line + '\n')
        loci_list = old_rec.loci_list
        positions.sort()
        positions.reverse()
        posSet = set()
        for pos in positions:
            del loci_list[pos]
            posSet.add(pos)
        for locus in loci_list:
            f.write(locus + '\n')
        l_parser = old_rec.get_individual()
        f.write('POP\n')
        while l_parser:
            if l_parser is True:
                f.write('POP\n')
            else:
                name, markers = l_parser
                f.write(name + ',')
                marker_pos = 0
                for marker in markers:
                    if marker_pos in posSet:
                        marker_pos += 1
                        continue
                    marker_pos += 1
                    f.write(' ')
                    for al in marker:
                        if al is None:
                            al = '0'
                        aStr = str(al)
                        while len(aStr) < 3:
                            aStr = ''.join(['0', aStr])
                        f.write(aStr)
                f.write('\n')
            l_parser = old_rec.get_individual()