def residue_dict(self, index):
    """Return a dict of lines in 'data' indexed by residue number or a nucleus.

        The nucleus should be given as the input argument in the same form as
        it appears in the xpk label line (H1, 15N for example)

        Parameters
        ----------
        index : str
            The nucleus to index data by.

        Returns
        -------
        resdict : dict
            Mappings of index nucleus to data line.

        Examples
        --------
        >>> from Bio.NMR.xpktools import Peaklist
        >>> peaklist = Peaklist('../Doc/examples/nmr/noed.xpk')
        >>> residue_d = peaklist.residue_dict('H1')
        >>> sorted(residue_d.keys())
        ['10', '3', '4', '5', '6', '7', '8', '9', 'maxres', 'minres']
        >>> residue_d['10']
        ['8  10.hn   7.663   0.021   0.010   ++   0.000   10.n   118.341   0.324   0.010   +E   0.000   10.n   118.476   0.324   0.010   +E   0.000  0.49840 0.49840 0']

        """
    maxres = -1
    minres = -1
    self.dict = {}
    for line in self.data:
        ind = XpkEntry(line, self.datalabels).fields[index + '.L']
        key = ind.split('.')[0]
        res = int(key)
        if maxres == -1:
            maxres = res
        if minres == -1:
            minres = res
        maxres = max([maxres, res])
        minres = min([minres, res])
        res = str(res)
        try:
            self.dict[res].append(line)
        except KeyError:
            self.dict[res] = [line]
    self.dict['maxres'] = maxres
    self.dict['minres'] = minres
    return self.dict