from Bio.Data import IUPACData
from typing import Dict, List, Optional
def list_ambiguous_codons(codons, ambiguous_nucleotide_values):
    """Extend a codon list to include all possible ambiguous codons.

    e.g.::

         ['TAG', 'TAA'] -> ['TAG', 'TAA', 'TAR']
         ['UAG', 'UGA'] -> ['UAG', 'UGA', 'URA']

    Note that ['TAG', 'TGA'] -> ['TAG', 'TGA'], this does not add 'TRR'
    (which could also mean 'TAA' or 'TGG').
    Thus only two more codons are added in the following:

    e.g.::

        ['TGA', 'TAA', 'TAG'] -> ['TGA', 'TAA', 'TAG', 'TRA', 'TAR']

    Returns a new (longer) list of codon strings.
    """
    c1_list = sorted((letter for letter, meanings in ambiguous_nucleotide_values.items() if {codon[0] for codon in codons}.issuperset(set(meanings))))
    c2_list = sorted((letter for letter, meanings in ambiguous_nucleotide_values.items() if {codon[1] for codon in codons}.issuperset(set(meanings))))
    c3_list = sorted((letter for letter, meanings in ambiguous_nucleotide_values.items() if {codon[2] for codon in codons}.issuperset(set(meanings))))
    candidates = []
    for c1 in c1_list:
        for c2 in c2_list:
            for c3 in c3_list:
                codon = c1 + c2 + c3
                if codon not in candidates and codon not in codons:
                    candidates.append(codon)
    answer = codons[:]
    for ambig_codon in candidates:
        wanted = True
        for codon in [c1 + c2 + c3 for c1 in ambiguous_nucleotide_values[ambig_codon[0]] for c2 in ambiguous_nucleotide_values[ambig_codon[1]] for c3 in ambiguous_nucleotide_values[ambig_codon[2]]]:
            if codon not in codons:
                wanted = False
                continue
        if wanted:
            answer.append(ambig_codon)
    return answer