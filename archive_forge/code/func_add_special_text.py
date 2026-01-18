from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def add_special_text(self, key, parent_keys, text):
    """Add a specific text to the graph.

        This is used to add a text which is not otherwise present in the
        versioned file. (eg. a WorkingTree injecting 'current:' into the
        graph to annotate the edited content.)

        :param key: The key to use to request this text be annotated
        :param parent_keys: The parents of this text
        :param text: A string containing the content of the text
        """
    self._parent_map[key] = parent_keys
    self._text_cache[key] = osutils.split_lines(text)
    self._heads_provider = None