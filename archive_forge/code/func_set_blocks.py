import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def set_blocks(self, blocks):
    """
        Assigns vectors in blocks

        Parameters
        ----------
        blocks: list
            list of numpy.ndarrays and/or BlockVectors

        Returns
        -------
        None

        """
    assert isinstance(blocks, list), 'blocks should be passed in ordered list'
    assert len(blocks) == self.nblocks, 'More blocks passed than allocated {} != {}'.format(len(blocks), self.nblocks)
    for idx, blk in enumerate(blocks):
        self.set_block(idx, blk)