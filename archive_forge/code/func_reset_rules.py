import configobj
from . import bedding, cmdline, errors, globbing, osutils
def reset_rules():
    global _per_user_searcher
    _per_user_searcher = _IniBasedRulesSearcher(rules_path())