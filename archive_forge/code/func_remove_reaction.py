import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def remove_reaction(self, reaction):
    """Remove a Reaction element from the pathway."""
    if not isinstance(reaction.id, int):
        raise TypeError(f'Node ID must be an integer, got {type(reaction.id)} ({reaction.id})')
    del self._reactions[reaction.id]