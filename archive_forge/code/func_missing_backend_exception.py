from collections import defaultdict
from ..core import Store
def missing_backend_exception(value, keyword, allowed):
    if value in OutputSettings.backend_list:
        raise ValueError(f'Backend {value!r} not available. Has it been loaded with the notebook_extension?')
    else:
        raise ValueError(f'Backend {value!r} does not exist')