from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
def mapping_with_config(self, config, key_dispatch):
    """Creates a new mapping object by applying a config object"""
    return ConfiguredEdits(self.simple_edits, self.cut_buffer_edits, self.awaiting_config, config, key_dispatch)