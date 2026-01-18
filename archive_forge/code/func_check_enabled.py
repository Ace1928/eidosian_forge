from stevedore import enabled
from stevedore.tests import utils
def check_enabled(ext):
    return ext.obj and ext.name == 't2'