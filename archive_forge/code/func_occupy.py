import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel4
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def occupy(self, position):
    """
        :return: Mark slot at ``position`` as occupied
        """
    self._slots[position] = True