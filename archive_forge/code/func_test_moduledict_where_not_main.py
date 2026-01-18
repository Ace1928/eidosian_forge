import os
import math
import dill as pickle
def test_moduledict_where_not_main():
    try:
        from . import test_moduledict
    except ImportError:
        import test_moduledict
    name = 'test_moduledict.py'
    if os.path.exists(name) and os.path.exists(name + 'c'):
        os.remove(name + 'c')
    if os.path.exists(name) and hasattr(test_moduledict, '__cached__') and os.path.exists(test_moduledict.__cached__):
        os.remove(getattr(test_moduledict, '__cached__'))
    if os.path.exists('__pycache__') and (not os.listdir('__pycache__')):
        os.removedirs('__pycache__')