from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
def set_chain_info(self, chain_id, chain_name, num_groups):
    """Set the chain information.

        :param chain_id: the asym chain id from mmCIF
        :param chain_name: the auth chain id from mmCIF
        :param num_groups: the number of groups this chain has

        """
    self.structure_builder.init_chain(chain_id=chain_name)
    if self.chain_index_to_type_map[self.chain_counter] == 'polymer':
        self.this_type = ' '
    elif self.chain_index_to_type_map[self.chain_counter] == 'non-polymer':
        self.this_type = 'H'
    elif self.chain_index_to_type_map[self.chain_counter] == 'water':
        self.this_type = 'W'
    self.chain_counter += 1