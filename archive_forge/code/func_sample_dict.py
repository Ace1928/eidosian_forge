import pytest
from ase.utils.filecache import MultiFileJSONCache, CombinedJSONCache, Locked
def sample_dict():
    return {'hello': [1, 2, 3], 'world': 'grumble'}