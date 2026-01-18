import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_invalid_formats(self):
    potentials = list()
    potentials.append(['human'])
    potentials.append('human')
    potentials.append(['human', 'star'])
    potentials.append([1, 2, 3, 4, 5, 5])
    potentials.append([[(0, 'string')]])
    for noCorpus in potentials:
        result = utils.is_corpus(noCorpus)
        expected = (False, noCorpus)
        self.assertEqual(expected, result)