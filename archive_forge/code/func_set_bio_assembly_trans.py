from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
def set_bio_assembly_trans(self, bio_assembly_index, input_chain_indices, input_transform):
    """Set the Bioassembly transformation information. A single bioassembly can have multiple transforms.

        :param bio_assembly_index: the integer index of the bioassembly
        :param input_chain_indices: the list of integer indices for the chains of this bioassembly
        :param input_transform: the list of doubles for  the transform of this bioassmbly transform.

        """