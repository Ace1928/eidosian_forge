import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def neighboring_mock(a, j):
    if a.alignment == (0, 3, 2):
        return {AlignmentInfo((0, 2, 2), None, None, None), AlignmentInfo((0, 1, 1), None, None, None)}
    elif a.alignment == (0, 2, 2):
        return {AlignmentInfo((0, 3, 3), None, None, None), AlignmentInfo((0, 4, 4), None, None, None)}
    return set()