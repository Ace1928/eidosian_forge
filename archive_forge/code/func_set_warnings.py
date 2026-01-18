import os
import sys
import warnings
from importlib.metadata import entry_points
import pytest
import networkx
@pytest.fixture(autouse=True)
def set_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='nx.nx_pydot')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='single_target_shortest_path_length will')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='shortest_path for all_pairs')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='\nforest_str is deprecated')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='\n\nrandom_tree')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='Edmonds has been deprecated')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='MultiDiGraph_EdgeKey has been deprecated')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='\n\nThe `normalized`')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='function `join` is deprecated')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='\n\nstrongly_connected_components_recursive')