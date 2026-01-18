import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_neighboring_returns_neighbors_with_pegged_alignment(self):
    a_info = AlignmentInfo((0, 3, 2), (None, 'des', 'œufs', 'verts'), ('UNUSED', 'green', 'eggs'), [[], [], [2], [1]])
    ibm_model = IBMModel([])
    neighbors = ibm_model.neighboring(a_info, 2)
    neighbor_alignments = set()
    for neighbor in neighbors:
        neighbor_alignments.add(neighbor.alignment)
    expected_alignments = {(0, 0, 2), (0, 1, 2), (0, 2, 2), (0, 3, 2)}
    self.assertEqual(neighbor_alignments, expected_alignments)