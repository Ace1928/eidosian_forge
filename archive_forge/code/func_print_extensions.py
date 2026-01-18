import importlib
from ..Qt import QT_LIB, QtGui
def print_extensions(ctx):
    extensions = sorted([ext.data().decode() for ext in ctx.extensions()])
    print('Extensions:')
    for ext in extensions:
        print(f'   {ext}')