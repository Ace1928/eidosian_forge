from Bio.Data import IUPACData
from typing import Dict, List, Optional
def list_possible_proteins(codon, forward_table, ambiguous_nucleotide_values):
    """Return all possible encoded amino acids for ambiguous codon."""
    c1, c2, c3 = codon
    x1 = ambiguous_nucleotide_values[c1]
    x2 = ambiguous_nucleotide_values[c2]
    x3 = ambiguous_nucleotide_values[c3]
    possible = {}
    stops = []
    for y1 in x1:
        for y2 in x2:
            for y3 in x3:
                try:
                    possible[forward_table[y1 + y2 + y3]] = 1
                except KeyError:
                    stops.append(y1 + y2 + y3)
    if stops:
        if possible:
            raise TranslationError(f'ambiguous codon {codon!r} codes for both proteins and stop codons')
        raise KeyError(codon)
    return list(possible)