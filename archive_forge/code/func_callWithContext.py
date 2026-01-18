from threading import local
from typing import Dict, Type
def callWithContext(self, ctx, func, *args, **kw):
    return self.currentContext().callWithContext(ctx, func, *args, **kw)