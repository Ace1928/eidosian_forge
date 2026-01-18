import math
import sys
from Bio import MissingPythonDependencyError
def get_row_positions(tree):
    positions = {taxon: 2 * idx for idx, taxon in enumerate(taxa)}

    def calc_row(clade):
        for subclade in clade:
            if subclade not in positions:
                calc_row(subclade)
        positions[clade] = (positions[clade.clades[0]] + positions[clade.clades[-1]]) // 2
    calc_row(tree.root)
    return positions