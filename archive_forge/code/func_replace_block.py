from collections import namedtuple
from collections.abc import Mapping
import copy
import inspect
import re
import textwrap
from statsmodels.tools.sm_exceptions import ParseError
def replace_block(self, block_name, block):
    """
        Parameters
        ----------
        block_name : str
            Name of the block to replace, e.g., 'Summary'.
        block : object
            The replacement block. The structure of the replacement block must
            match how the block is stored by NumpyDocString.
        """
    if self._docstring is None:
        return
    block_name = ' '.join(map(str.capitalize, block_name.split(' ')))
    if block_name not in self._ds:
        raise ValueError('{} is not a block in the docstring'.format(block_name))
    if not isinstance(block, list) and isinstance(self._ds[block_name], list):
        block = [block]
    self._ds[block_name] = block