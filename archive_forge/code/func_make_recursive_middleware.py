import warnings
from io import StringIO
def make_recursive_middleware(app, global_conf):
    return RecursiveMiddleware(app)