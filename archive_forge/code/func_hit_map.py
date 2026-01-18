from copy import deepcopy
from itertools import chain
from Bio.SearchIO._utils import optionalcascade
from ._base import _BaseSearchObject
from .hit import Hit
def hit_map(self, func=None):
    """Create new QueryResult object, mapping the given function to its Hits.

        :param func: map function
        :type func: callable, accepts Hit, returns Hit

        Here is an example of using ``hit_map`` with a function that discards all
        HSPs in a Hit except for the first one::

            >>> from Bio import SearchIO
            >>> qresult = next(SearchIO.parse('Blast/mirna.xml', 'blast-xml'))
            >>> print(qresult[:8])
            Program: blastn (2.2.27+)
              Query: 33211 (61)
                     mir_1
             Target: refseq_rna
               Hits: ----  -----  ----------------------------------------------------------
                        #  # HSP  ID + description
                     ----  -----  ----------------------------------------------------------
                        0      1  gi|262205317|ref|NR_030195.1|  Homo sapiens microRNA 52...
                        1      1  gi|301171311|ref|NR_035856.1|  Pan troglodytes microRNA...
                        2      1  gi|270133242|ref|NR_032573.1|  Macaca mulatta microRNA ...
                        3      2  gi|301171322|ref|NR_035857.1|  Pan troglodytes microRNA...
                        4      1  gi|301171267|ref|NR_035851.1|  Pan troglodytes microRNA...
                        5      2  gi|262205330|ref|NR_030198.1|  Homo sapiens microRNA 52...
                        6      1  gi|262205302|ref|NR_030191.1|  Homo sapiens microRNA 51...
                        7      1  gi|301171259|ref|NR_035850.1|  Pan troglodytes microRNA...

            >>> top_hsp = lambda hit: hit[:1]
            >>> mapped_qresult = qresult.hit_map(top_hsp)
            >>> print(mapped_qresult[:8])
            Program: blastn (2.2.27+)
              Query: 33211 (61)
                     mir_1
             Target: refseq_rna
               Hits: ----  -----  ----------------------------------------------------------
                        #  # HSP  ID + description
                     ----  -----  ----------------------------------------------------------
                        0      1  gi|262205317|ref|NR_030195.1|  Homo sapiens microRNA 52...
                        1      1  gi|301171311|ref|NR_035856.1|  Pan troglodytes microRNA...
                        2      1  gi|270133242|ref|NR_032573.1|  Macaca mulatta microRNA ...
                        3      1  gi|301171322|ref|NR_035857.1|  Pan troglodytes microRNA...
                        4      1  gi|301171267|ref|NR_035851.1|  Pan troglodytes microRNA...
                        5      1  gi|262205330|ref|NR_030198.1|  Homo sapiens microRNA 52...
                        6      1  gi|262205302|ref|NR_030191.1|  Homo sapiens microRNA 51...
                        7      1  gi|301171259|ref|NR_035850.1|  Pan troglodytes microRNA...

        """
    hits = [deepcopy(hit) for hit in self.hits]
    if func is not None:
        hits = [func(x) for x in hits]
    obj = self.__class__(hits, self.id, self._hit_key_function)
    self._transfer_attrs(obj)
    return obj