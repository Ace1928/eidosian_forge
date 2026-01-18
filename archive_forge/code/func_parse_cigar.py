import re
import enum
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def parse_cigar(self, cigar):
    """Parse a CIGAR string and return alignment coordinates.

        A CIGAR string, as defined by the SAM Sequence Alignment/Map format,
        describes a sequence alignment as a series of lengths and operation
        (alignment/insertion/deletion) codes.
        """
    target_coordinates = []
    query_coordinates = []
    target_coordinate = 0
    query_coordinate = 0
    target_coordinates.append(target_coordinate)
    query_coordinates.append(query_coordinate)
    state = State.NONE
    tokens = re.findall('(M|D|I|\\d+)', cigar)
    for length, operation in zip(tokens[::2], tokens[1::2]):
        length = int(length)
        if operation == 'M':
            target_coordinate += length
            query_coordinate += length
        elif operation == 'I':
            target_coordinate += length
        elif operation == 'D':
            query_coordinate += length
        target_coordinates.append(target_coordinate)
        query_coordinates.append(query_coordinate)
    coordinates = np.array([target_coordinates, query_coordinates])
    return coordinates