import decorator
from moviepy.tools import cvsecs
def convert_to_seconds(varnames):
    """Converts the specified variables to seconds"""
    return preprocess_args(cvsecs, varnames)