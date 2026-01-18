import sys
import builtins
def monkeypatch_quit():
    if 'site' in sys.modules:
        resetquit(builtins)