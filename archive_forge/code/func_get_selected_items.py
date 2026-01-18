import configobj
from . import bedding, cmdline, errors, globbing, osutils
def get_selected_items(self, path, names):
    """See _RulesSearcher.get_selected_items."""
    for searcher in self.searchers:
        result = searcher.get_selected_items(path, names)
        if result:
            return result
    return ()